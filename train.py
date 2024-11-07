import datetime
import shutil
import time
import warnings
from pathlib import Path

import math
import torch
import torch.distributed as dist
from timm.optim import create_optimizer_v2
from timm.utils import AverageMeter
from torch.cuda.amp import GradScaler, autocast

from data import build_loader
from model import build_ddpm, build_classifier, EMAHelper
from utils import create_scheduler, optimizer_kwargs, save_checkpoint, set_weight_decay, \
    multi_process_setup, distributed_training, auto_resume, generate_xt, backward, get_alpha_and_beta, cond_fn, \
    check_path_is_file_or_dir, concat_all_gather, get_img_from_dataloader_iter

warnings.filterwarnings("ignore")


def main(local_rank, config):
    config, logger = multi_process_setup(local_rank)

    # build datasets, dataset_loader
    dsets, dset_loaders = build_loader(config)

    # build the model
    model = build_ddpm(config, logger, with_adapter=(not config.model.ddpm.pretrain))
    model.cuda()

    classifier = build_classifier(config, logger)
    classifier.eval()
    classifier.cuda()
    if dist.get_rank() == 0:
        logger.info("==============>ddpm_model....................")
        logger.info(str(model))

    # build the optimizer
    params_list = set_weight_decay(model)
    optimizer = create_optimizer_v2(params_list, **optimizer_kwargs(config))

    # learning rate scheduler
    lr_scheduler, num_iteration = create_scheduler(config, optimizer)

    start_iteration = 0

    # Mixed-Precision and distributed training
    models, optimizer = distributed_training([model, classifier], optimizer, local_rank, logger)
    model = models[0]
    classifier = models[1]
    scaler = GradScaler()

    if config.model.ddpm.pretrain or not config.tl.classifier:
        # if not pre-train ddpm model or not use classifier, delete the classifier model
        del classifier
        classifier = None

    model_without_ddp = model.module

    if config.model.ddpm.ema:
        ema_helper = EMAHelper(mu=config.model.ddpm.ema_rate)
        ema_helper.register(model)
    else:
        ema_helper = None

    if config.auto_resume and check_path_is_file_or_dir(Path(config.output).joinpath("latest.pt")):
        start_iteration = auto_resume(config, model_without_ddp, optimizer, lr_scheduler, scaler, logger, ema_helper,"latest.pt")

    if ema_helper:
        ema_helper.to_cuda(model)

    lr_scheduler.step(start_iteration)
    dset_loaders["train"].sampler.set_epoch(start_iteration)

    s_time = time.time()
    logger.info(f"==============>Start train model....................")
    training([model, classifier], dset_loaders, optimizer, lr_scheduler, ema_helper, scaler, logger, config, start_iteration)

    end_time = time.time() - s_time
    logger.info(f"Training takes {datetime.timedelta(seconds=int(end_time))}")
    logger.info("Done!")


def training(models, dset_loaders, optimizer, lr_scheduler, ema_helper, scaler, logger, config, start_iteration=0):
    model = models[0]
    model.train()
    model_without_ddp = model.module

    if models[1] is not None:
        classifier = models[1]
        classifier.eval()
    else:
        classifier = None

    optimizer.zero_grad()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    num_timesteps = config.dm.num_diffusion_timesteps

    data_iter = iter(dset_loaders['train'])
    iterations = config.train.iteration
    a, betas = get_alpha_and_beta(config)

    end = time.time()

    omega = config.tl.ad_omega
    ad_num_iter = config.tl.ad_num_iter

    criterion = torch.nn.MSELoss(reduction='none').cuda()

    c = config.tl.c
    variance = 1.0 - a
    variance = variance.cuda()

    torch_type = torch.float16 if config.amp_opt_level != "O0" else torch.float32

    x0 = get_img_from_dataloader_iter(dset_loaders['train'], data_iter, 1)
    x0 = x0.cuda(non_blocking=True)

    with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
        with autocast(dtype=torch_type):
            for idx in range(start_iteration, iterations):
                if idx >= 321:
                    break
                if idx % config.reset_average_meter == 0:
                    loss_meter.reset()
                    norm_meter.reset()
                    logger.info("\t")

                B = x0.shape[0]

                e, xt, t = generate_xt(x0, a, B, num_timesteps)

                data_time.update(time.time() - end)

                if config.tl.ad_train and not config.model.ddpm.pretrain:
                    # min-max adversarial training
                    with torch.enable_grad():
                        for _ in range(ad_num_iter):
                            model.zero_grad()
                            output = model(xt, t)
                            loss = criterion(e, output).sum(dim=[1, 2, 3]).mean(dim=0)
                            loss.backward()
                            e_gard = e.grad.detach()
                            e = e + omega * e_gard
                            e_all = concat_all_gather(e)
                            e_all = (e_all - e_all.mean((1, 2, 3), keepdim=True)) / e_all.std((1, 2, 3), keepdim=True)
                            e = e_all[config.local_rank * B:(config.local_rank + 1) * B, ]
                            del e_all
                            e = e.cuda()
                            e, xt, t = generate_xt(x0, a, B, num_timesteps, e.detach())
                            e, xt = e.detach().requires_grad_(True), xt.detach().requires_grad_(True)
                            torch.cuda.synchronize()

                output = model(xt, t.float())

                # use the classifier to guide train loss
                if classifier is not None:
                    y = torch.zeros(B, dtype=torch.long).cuda()
                    target_source = cond_fn(classifier, xt, t.float(), y).detach()
                else:
                    target_source = 0.0

                variance_t = variance.index_select(0, t.long()).view(-1, 1, 1, 1)

                loss = criterion(e - c * variance_t * target_source, output).sum(dim=[1, 2, 3]).mean(dim=0)
                grad_norm = backward(model, loss, optimizer, scaler, config)

                if ema_helper is not None:
                    ema_helper.update(model)

                torch.cuda.synchronize()
                loss_meter.update(loss.item(), B)
                if not math.isnan(grad_norm) and not math.isinf(grad_norm):
                    norm_meter.update(grad_norm)
                batch_time.update(time.time() - end)
                end = time.time()

                del loss, xt, e, output, target_source

                lr = optimizer.param_groups[0]['lr']

                lr_scheduler.step_update(num_updates=idx + 1, metric=loss_meter.avg)
                lr_scheduler.step(idx + 1)

                if (idx % config.save_freq == 0 or idx == (iterations - 1)):
                    save_checkpoint(config=config,
                                    model=model_without_ddp,
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler,
                                    iteration=idx + 1,
                                    scaler=scaler,
                                    logger=logger,
                                    ema=ema_helper,
                                    name=f"latest.pt")
                    if config.local_rank == 0:
                        shutil.copytree(Path(config.output).joinpath(f"latest.pt"),
                                        Path(config.output).joinpath(f"iteration_{idx}.pt"))

                if idx % config.log_freq == 0:
                    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                    etas = batch_time.avg * (iterations - idx)
                    logger.info(
                        f'Train: [{idx}/{iterations}]\t'
                        f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.8f}\t'
                        f'data time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                        f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                        f'mem {memory_used:.0f}MB')


if __name__ == '__main__':
    ngpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)
