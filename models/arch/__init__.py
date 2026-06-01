# Add your custom network here
from .default import DRNet
import torch.nn as nn

from .cascade_errnet import CascadeERRNet


def basenet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, bottom_kernel_size=1, **kwargs)


def errnet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True, **kwargs)

def cascade_errnet(in_channels, out_channels, **kwargs):
    return CascadeERRNet(in_channels, out_channels)
