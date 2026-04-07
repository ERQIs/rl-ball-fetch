import torch
import torch.nn as nn


def get_loss():
    return nn.SmoothL1Loss()
