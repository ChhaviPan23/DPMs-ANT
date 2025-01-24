import datetime
import shutil
import time
from pathlib import Path
import os
import numpy as np
import math
import torch
import torch.distributed as dist
from timm.optim import create_optimizer_v2
from timm.utils import AverageMeter
from torch.cuda.amp import GradScaler, autocast
import PIL
from data import build_loader
from model import build_classifier
from utils import create_scheduler, auto_resume, optimizer_kwargs, \
    set_weight_decay, multi_process_setup, distributed_training, generate_xt, backward, \
    save_checkpoint, get_alpha_and_beta, get_img_from_dataloader_iter, check_path_is_file_or_dir


def main(local_rank):
    config, logger = multi_process_setup(local_rank)

    # build datasets, dataset_loaders
    dsets, dset_loaders = build_loader(config)

    # build the classifier model
    classifier = build_classifier(config, logger)
    classifier.cuda()

    scaler = GradScaler()

    params_list = set_weight_decay(classifier)
    optimizer = create_optimizer_v2(params_list, **optimizer_kwargs(config))

    # learning rate scheduler
    lr_scheduler, num_iteration = create_scheduler(config, optimizer)

    start_iteration = 0

    # Mixed-Precision and distributed training
    classifier, optimizer = distributed_training([classifier], optimizer, local_rank, logger)
    classifier = classifier[0]
    classifier_without_ddp = classifier.module

    if config.auto_resume and check_path_is_file_or_dir(Path(config.output).joinpath("latest.pt")):
        start_iteration = auto_resume(config, classifier_without_ddp, optimizer, lr_scheduler, scaler, logger, None,
                                      "latest.pt")

    lr_scheduler.step(start_iteration)
    dset_loaders["train"].sampler.set_epoch(start_iteration)
    iterations = config.train.iteration


    s_time = time.time()
    logger.info(f"==============>Start train model....................")
    for epoch in range(start_iteration, iterations):
        dset_loaders["train"].sampler.set_epoch(epoch)
        if not config.model.classifier.pretrain:
            dset_loaders["train_source"].sampler.set_epoch(epoch)
        training(classifier, dset_loaders, optimizer, scaler, epoch, config, logger)

        if (epoch % config.save_freq == 0 or epoch == (iterations - 1)):
            save_checkpoint(config=config,
                            model=classifier_without_ddp,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            iteration=epoch + 1, scaler=scaler, logger=logger, ema=None,
                            name=f"latest.pt")

            if dist.get_rank() == 0:
                shutil.copy(Path(config.output).joinpath(f"latest.pt"),
                                Path(config.output).joinpath(f"iteration_{epoch}.pt"))

        lr_scheduler.step(epoch + 1)

    end_time = time.time() - s_time
    logger.info(f"Training takes {datetime.timedelta(seconds=int(end_time))}")
    logger.info("Done!")


def training(model, dset_loaders, optimizer, scaler, epoch, config, logger):
    model.train()
    num_timesteps = config.dm.num_diffusion_timesteps

    alphas, betas = get_alpha_and_beta(config)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()

    target_dataloader = dset_loaders["train"]
    target_iter = iter(target_dataloader)

    source_dataloader = None
    source_iter = None

    iters_per_epoch = len(target_dataloader)
    with autocast(dtype=torch.float16):
        for idx in range(iters_per_epoch):
            if not config.model.classifier.pretrain:
                # finetune classifier model on the source and target datasets
                if idx == 0:
                    source_dataloader = dset_loaders["train_source"]
                    source_iter = iter(source_dataloader)
                source_img = get_img_from_dataloader_iter(source_dataloader, source_iter, epoch)
                target_img = get_img_from_dataloader_iter(target_dataloader, target_iter, epoch)
                x0 = torch.cat([source_img, target_img], dim=0)
                labels = torch.cat([torch.ones(source_img.shape[0], dtype=torch.long),
                                    torch.zeros(target_img.shape[0], dtype=torch.long)], dim=0)
            else:
                # pre-train classifier model on the ImageNet datasets
                x0, labels = get_img_from_dataloader_iter(target_dataloader, target_iter, epoch)

            B = x0.shape[0]

            x0, labels = x0.cuda(non_blocking=True), labels.cuda(non_blocking=True)

            e, xt, t = generate_xt(x0, alphas, B, num_timesteps)
            data_time.update(time.time() - end)

            outputs = model(xt, t)
            loss = criterion(outputs, labels)

            grad_norm = backward(model, loss, optimizer, scaler, config)

            torch.cuda.synchronize()
            # record loss
            loss_meter.update(loss.item(), B)
            lr = optimizer.param_groups[0]['lr']
            if not math.isnan(grad_norm) and not math.isinf(grad_norm):
                norm_meter.update(grad_norm)
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % config.log_freq == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (iters_per_epoch - idx)
                logger.info(
                    f'Train: [{epoch}/{config.train.iteration}]: [{idx}/{iters_per_epoch}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.8f}\t'
                    f'data time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(f"==============>EPOCH {epoch}....................")
    logger.info(f'EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}\t'
                f'loss ({loss_meter.avg:.4f})\t'
                f'grad_norm ({norm_meter.avg:.4f})\t')


if __name__ == '__main__':
    ngpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)
