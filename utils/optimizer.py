def optimizer_kwargs(config):
    """ cfg/argparse to kwargs helper
    Convert optimizer args in argparse args or cfg like object to keyword args for updated create fn.
    """
    kwargs = dict(
            opt=config.opt.name,
            lr=config.train.lr,
            weight_decay=config.opt.weight_decay,
            momentum=config.opt.momentum)
    if config.opt.eps is not None:
        kwargs['eps'] = config.opt.eps
    if config.opt.betas is not None and 'adam' in config.opt.name.lower():
        kwargs['betas'] = config.opt.betas
    if config.opt.layer_decay is not None:
        kwargs['layer_decay'] = config.opt.layer_decay
    return kwargs


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
