import torch
from torch import nn

from utils import get_obj_from_str


class LatentDiffusion(nn.Module):
    def __init__(self,
                 first_stage_config,
                 ddpm_config,
                 scale_factor=1.0,
                 scale_by_std=False):
        super().__init__()
        self.scale_by_std = scale_by_std
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.first_stage_model = get_obj_from_str(first_stage_config.target)(**first_stage_config.params)
        self.model = get_obj_from_str(ddpm_config.target)(**ddpm_config.params)

        # freeze first stage model
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
