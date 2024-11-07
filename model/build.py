from pathlib import Path

import torch
from torch.nn import Conv1d

from model.guided_diffusion import EncoderUNetModel
from model.ldm.latent_diffusion import LatentDiffusion
from utils import get_obj_from_str
from yacs.config import CfgNode as CN


def build_lmd(config, logger, with_adapter=True):
    # attention_ds = []
    # for res in config.model.ddpm.attn_resolutions:
    #     attention_ds.append(config.data.img_size // int(res))
    ddpm_config = CN()
    ddpm_config.target = "model.guided_diffusion.unet.UNetModel"
    ddpm_config.params = CN()
    ddpm_config.params.image_size = config.model.ddpm.img_size
    ddpm_config.params.in_channels = config.model.ddpm.in_channels
    ddpm_config.params.model_channels = config.model.ddpm.ch
    ddpm_config.params.out_channels = config.model.ddpm.out_ch
    ddpm_config.params.num_res_blocks =  config.model.ddpm.num_res_blocks
    ddpm_config.params.attention_resolutions = config.model.ddpm.attn_resolutions
    ddpm_config.params.dropout = config.model.ddpm.dropout
    ddpm_config.params.channel_mult = tuple(config.model.ddpm.ch_mult)
    ddpm_config.params.num_classes = None
    ddpm_config.params.use_checkpoint = True
    ddpm_config.params.use_fp16 = config.amp_opt_level != "O0"
    ddpm_config.params.num_heads = config.model.ddpm.num_heads
    ddpm_config.params.num_head_channels = config.model.ddpm.num_head_channels
    ddpm_config.params.num_heads_upsample = -1
    ddpm_config.params.use_scale_shift_norm = config.model.ddpm.use_scale_shift_norm
    ddpm_config.params.resblock_updown = config.model.ddpm.resamp_with_conv
    ddpm_config.params.use_new_attention_order = False
    ddpm_config.params.with_adapter = with_adapter
    ddpm_config.params.adapter_dim = config.model.adapter.dim
    ddpm_config.params.adapter_patch_size = config.model.adapter.patch_size
    ddpm_config.params.adapter_num_heads = config.model.adapter.num_heads
    ddpm_config.params.adapter_qkv_bias = config.model.adapter.qkv_bias
    ddpm_config.params.adapter_drop = config.model.adapter.drop

    model = LatentDiffusion(first_stage_config=config.model.first_stage_config,
                            ddpm_config=ddpm_config,
                            scale_by_std=config.model.scale_by_std)
    # path = Path("/home/xywang/tmp/model.ckpt")
    # from torch import load
    # sd = load(path, map_location='cpu')
    # if "state_dict" in list(sd.keys()):
    #     sd = sd["state_dict"]
    #
    # ckpt = {}
    # for k in list(sd.keys()):
    #     if k.startswith("first_stage_model."):
    #         ckpt[str(k).replace("first_stage_model.", "")] = sd[k]
    # missing, unexpected = model.first_stage_model.load_state_dict(ckpt, strict=False)
    # print("missing", missing)
    # print("unexpected", unexpected)
    #
    # ckpt = {}
    # for k in list(sd.keys()):
    #     if k.startswith("model.diffusion_model."):
    #         ckpt[str(k).replace("model.diffusion_model.", "")] = \
    #             sd[f'model_ema.diffusion_model{k.replace("model.diffusion_model.", "").replace(".", "")}']
    # missing, unexpected = model.model.load_state_dict(ckpt, strict=False)
    # print("missing", missing)
    # print("unexpected", unexpected)
    # outpath = "/home/xywang/tmp/ckpt/CHURCH-LDM.ckpt"
    #
    # model.scale_factor = sd["scale_factor"]
    # print(model.scale_factor)
    #
    # torch.save(model.state_dict(), outpath)
    # exit()

    if config.model.ddpm.initial_checkpoint != "":
        load_initial_checkpoint(model, config.model.ddpm.initial_checkpoint, logger)
    return model


def build_ddpm(config, logger, with_adapter=True):
    if "DDPM.diffusion.Model" in config.model.ddpm.target:
        # use the original DDPM from https://github.com/pesser/pytorch_diffusion/tree/master/pytorch_diffusion
        model_kwargs = {"ch": config.model.ddpm.ch,
                        "in_channels": config.model.ddpm.in_channels,
                        "out_ch": config.model.ddpm.out_ch,
                        "ch_mult": tuple(config.model.ddpm.ch_mult),
                        "num_res_blocks": config.model.ddpm.num_res_blocks,
                        "attn_resolutions": config.model.ddpm.attn_resolutions,
                        "dropout": config.model.ddpm.dropout,
                        "resamp_with_conv": config.model.ddpm.resamp_with_conv,
                        "model_type": config.model.ddpm.var_type,
                        "img_size": config.data.img_size,
                        "num_timesteps": config.dm.num_diffusion_timesteps,
                        "with_adapter": with_adapter,
                        "adapter_dim": config.model.adapter.dim,
                        "adapter_patch_size": config.model.adapter.patch_size,
                        "adapter_num_heads": config.model.adapter.num_heads,
                        "adapter_qkv_bias": config.model.adapter.qkv_bias,
                        "adapter_drop": config.model.adapter.drop}
    elif "guided_diffusion.unet.UNetModel" in config.model.ddpm.target:
        # use the guided diffusion DDPM from
        # https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py
        attention_ds = []
        for res in config.model.ddpm.attn_resolutions:
            attention_ds.append(config.data.img_size // int(res))

        model_kwargs = {"image_size": config.data.img_size,
                        "in_channels": config.model.ddpm.in_channels,
                        "model_channels": config.model.ddpm.ch,
                        "out_channels": config.model.ddpm.out_ch,
                        "num_res_blocks": config.model.ddpm.num_res_blocks,
                        "attention_resolutions": attention_ds,
                        "dropout": config.model.ddpm.dropout,
                        "channel_mult": tuple(config.model.ddpm.ch_mult),
                        "num_classes": None,
                        "use_checkpoint": True,
                        "use_fp16": config.amp_opt_level != "O0",
                        "num_heads": config.model.ddpm.num_heads,
                        "num_head_channels": config.model.ddpm.num_head_channels,
                        "num_heads_upsample": -1,
                        "use_scale_shift_norm": True,
                        "resblock_updown": True,
                        "use_new_attention_order": False,
                        "with_adapter": with_adapter,
                        "adapter_dim": config.model.adapter.dim,
                        "adapter_patch_size": config.model.adapter.patch_size,
                        "adapter_num_heads": config.model.adapter.num_heads,
                        "adapter_qkv_bias": config.model.adapter.qkv_bias,
                        "adapter_drop": config.model.adapter.drop}
    else:
        raise Exception(f"It not supports the model {config.model.ddpm.target}!")

    model = get_obj_from_str(config.model.ddpm.target)(**model_kwargs)

    if config.model.ddpm.initial_checkpoint != "":
        load_initial_checkpoint(model, config.model.ddpm.initial_checkpoint, logger)

    return model


def build_classifier(config, logger=None):
    # use the guided diffusion classifier from
    # https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py
    attention_ds = []
    for res in config.model.classifier.attn_resolutions:
        attention_ds.append(config.model.ddpm.img_size // int(res))

    cfg = dict(
        image_size=config.model.ddpm.img_size,
        in_channels=config.model.classifier.in_channels,
        model_channels=config.model.classifier.ch,
        out_channels=1000,
        num_res_blocks=config.model.classifier.num_res_blocks,
        attention_resolutions=attention_ds,
        channel_mult=tuple(config.model.classifier.ch_mult),
        num_head_channels=64,
        use_scale_shift_norm=True,  # False
        resblock_updown=True,  # False
        pool="attention",
        use_checkpoint=True,
    )

    model = EncoderUNetModel(**cfg)

    if config.model.classifier.initial_checkpoint != "":
        try:
            load_initial_checkpoint(model, config.model.classifier.initial_checkpoint, logger)
            model.out[2].c_proj = Conv1d(model.out[2].c_proj.in_channels, 2, 1)
        except:
            model.out[2].c_proj = Conv1d(model.out[2].c_proj.in_channels, 2, 1)
            load_initial_checkpoint(model, config.model.classifier.initial_checkpoint, logger)


    # from hfai.checkpoint import load
    # load_path = "results/Classifier_FFHQ2SUM_100_1e-4/latest.pt"
    # ckpt = load(load_path, map_location='cpu')
    #
    # msg = model.load_state_dict(ckpt['model'], strict=False)
    # print(msg)
    # torch.save(model.state_dict(), "/home/xywang/tmp/ckpt/FFHQ2SUN_classifier_64_100.pt")
    # exit()

    return model


def load_initial_checkpoint(model, initial_checkpoint, logger):
    logger.info(f"======> Loading pre-trained model {initial_checkpoint}")
    msg = model.load_state_dict(torch.load(initial_checkpoint), strict=False)
    logger.info(msg)
    logger.info(f"======> Success loading pre-trained model {initial_checkpoint}")
