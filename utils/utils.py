import importlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from configs.config import parse_option
from utils.logger import create_logger

def save(filepath, model, optimizer, others=None):
    # Prepare the state dictionary
checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if others:
        checkpoint['others'] = others

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load(filepath, map_location = 'cpu'):
    filepath = Path(filepath)
    checkpoint = torch.load(filepath, map_location=map_location)
    print(f"Checkpoint loaded from {filepath}.")
    return checkpoint


def multi_process_setup(local_rank):
    """
    setup for the multiprocessing running
    :param local_rank: the rank of current node
    :return: config and logger
    """
    # load hyperparameter
    _, config = parse_option()

    Path(config.output).mkdir(parents=True, exist_ok=True)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    # Multi-node communication
    ip = os.environ['MASTER_IP']
    port = config.port if config.port != -1 else os.environ['MASTER_PORT']
    hosts = int(os.environ['WORLD_SIZE'])  # number of node
    rank = int(os.environ['RANK'])  # rank of current node
    gpus = torch.cuda.device_count()  # Number of GPUs per node

    # world_size is the number of global GPU, rank is the global index of current GPU
    dist.init_process_group(backend='nccl', init_method=f'tcp://{ip}:{port}', world_size=hosts * gpus,
                            rank=rank * gpus + local_rank)
    torch.cuda.set_device(local_rank)

    # linear scale the learning rate according to total batch size, may not be optimal
    lr = config.train.lr
    warmup_lr = config.train.lr * config.train.warmup_lr_rate
    min_lr = config.train.lr * config.train.min_lr_rate

    config.defrost()
    config.train.lr = lr
    config.train.warmup_lr = warmup_lr
    config.train.min_lr = min_lr
    config.local_rank = local_rank
    config.freeze()

    # fix the random seed
    seed = config.seed + local_rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # create the logger
    logger_name = f"dic_diff"
    logger = create_logger(output_dir=config.output, dist_rank=local_rank, name=logger_name)

    if dist.get_rank() == 0:
        path = Path(config.output).joinpath("config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(config)
        logger.info(f"Full config saved to {path}")

    return config, logger


def distributed_training(models, optimizer, local_rank, logger):
    """
    setup the DDP with or without apex with one model
    :param model:
    :param optimizer:
    :param local_rank:
    :param logger:
    :return:
    """
    model_list = list()
    for model in models:
        model = DistributedDataParallel(model,
                                        device_ids=[local_rank],
                                        broadcast_buffers=False,
                                        static_graph=True)
        model_list.append(model)
    models = model_list
    logger.info("Using native Torch DistributedDataParallel.")
    return models, optimizer


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def backward(model, loss, optimizer, scaler, config):
    """
    :param model:
    :param loss:
    :param optimizer:
    :param scaler:
    :param config:
    :return:
    """
    # compute gradient and do step
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    # Unscales the gradients of optimizer's assigned params in-place
    scaler.unscale_(optimizer)
    if config.opt.clip_grad:
        grad_norm = torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                                   config.opt.clip_grad)
    else:
        grad_norm = get_grad_norm(filter(lambda p: p.requires_grad, model.parameters()))
    scaler.step(optimizer)
    scaler.update()
    return grad_norm


def get_img_from_dataloader_iter(dataloader, data_loader_iter, idx):
    try:
        img = next(data_loader_iter)
    except:
        dataloader.sampler.set_epoch(idx)
        data_loader_iter = iter(dataloader)
        img = next(data_loader_iter)
    return img


def check_path_is_file_or_dir(path: Path) -> bool:
    return path.is_dir() or path.is_file()


def load_checkpoint(config, model, optimizer, lr_scheduler, logger, scaler=None, ema=None, name='latest.pt'):
    """
    load the checkpoint for auto resume
    :param config:
    :param model:
    :param optimizer:
    :param lr_scheduler:
    :param logger:
    :param scaler:
    :param name:
    :return:
    """
    logger.info(f"==============> Resuming form {Path(config.output).joinpath(name)}....................")
    ckpt = load(Path(config.output).joinpath(name), map_location='cpu')

    model.load_state_dict(ckpt['model'], strict=False)

    start_epoch = 0

    if ckpt['iteration'] is not None:
        start_epoch = ckpt['iteration']
    if ckpt['optimizer'] is not None and optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    if ckpt['lr_scheduler'] is not None and lr_scheduler is not None:
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
    if scaler is not None and ckpt['scaler'] is not None:
        scaler.load_state_dict(ckpt['scaler'])
    if ema is not None and ckpt['ema'] is not None:
        ema.load_state_dict(ckpt['ema'])
    logger.info(f"==============> success resume form to {Path(config.output).joinpath(name)}....................")

    return start_epoch


def save_checkpoint(config, model, optimizer, lr_scheduler, iteration, scaler=None, logger=None, ema=None,
                    name="latest.pt"):
    """
    save the checkpoint for the auto resume and least model
    :param config:
    :param model:
    :param optimizer:
    :param lr_scheduler:
    :param iteration:
    :param scaler:
    :param logger:
    :param ema:
    :param name:
    :return:
    """
    logger.info(f"==============> Saving to {Path(config.output).joinpath(name)}....................")
    save_state = {'lr_scheduler': lr_scheduler.state_dict(),
                  'iteration': iteration,
                  'scaler': None,
                  'ema': None,
                  'config': config}

    if scaler is not None:
        save_state['scaler'] = scaler.state_dict()

    if ema is not None:
        save_state['ema'] = ema.state_dict()

    out_path = Path(config.output).joinpath(name)
    save(out_path, model, optimizer, others=save_state)
    logger.info(f"==============> success save to {Path(config.output).joinpath(name)}....................")


def auto_resume(config, model_without_ddp, optimizer, lr_scheduler, scaler, logger, ema, name):
    start_iteration = int(load_checkpoint(config=config,
                                          model=model_without_ddp,
                                          optimizer=optimizer,
                                          lr_scheduler=lr_scheduler,
                                          scaler=scaler,
                                          logger=logger,
                                          ema=ema,
                                          name=name))
    return start_iteration
