"""Learning rate scheduling policies."""

__author__ = 'Erdene-Ochir Tuguldur'
__all__ = ['noam_v1', 'cosine_annealing']

import math


def noam_v1(lr, step, epoch, min_lr, warmup_steps, total_steps):
    # https: // github.com / tensorflow / tensor2tensor / issues / 280
    new_lr = lr * warmup_steps ** 0.5 * min((step + 1) * warmup_steps ** -1.5, (step + 1) ** -0.5)
    return new_lr


def cosine_annealing(lr, step, epoch, min_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        return lr * (step + 1) / (warmup_steps + 1)
    elif step >= total_steps:
        return min_lr
    else:
        mult = 0.5 * (1 + math.cos(math.pi * step / total_steps))
        out_lr = (lr - min_lr) * mult + min_lr
        return out_lr
