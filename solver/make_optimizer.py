import torch

def make_optimizer(Cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = Cfg.SOLVER.BASE_LR
        weight_decay = Cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = Cfg.SOLVER.BASE_LR * Cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = Cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if Cfg.SOLVER.OPTIMIZER == 'SGD':
        optimizer = getattr(torch.optim, Cfg.SOLVER.OPTIMIZER)(params, momentum=Cfg.SOLVER.MOMENTUM)

    else:
        optimizer = getattr(torch.optim, Cfg.SOLVER.OPTIMIZER)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=Cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center