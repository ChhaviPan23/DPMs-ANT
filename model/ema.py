import torch.nn as nn


class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().cuda()

    def update(self, module):
        module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data.cuda())

    def state_dict(self):
        return self.shadow

    def to_cuda(self, module):
        module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = self.shadow[name].data.cuda()

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
