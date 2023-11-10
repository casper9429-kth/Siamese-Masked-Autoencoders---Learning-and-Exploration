# This code is based on or includes code from the Facebook Research MAE repository.
# Original source: https://github.com/facebookresearch/mae
# Licensed under CC-BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International License).

import math

def adjust_learning_rate(optimizer, epoch, params):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < params["warmup_epochs"]:
        lr = params["lr"] * epoch / params["warmup_epochs"] 
    else:
        lr = params["min_lr"] + (params["lr"] - params["min_lr"]) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - params["warmup_epochs"]) / (params["epochs"] - params["warmup_epochs"])))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr