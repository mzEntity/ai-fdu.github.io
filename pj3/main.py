from nets.RGBTCCNet import *
import torch.nn as nn
from nets.pvt_v2 import pvt_v2_b3
from nets.transformer_decoder_noPos import transfmrerDecoder
from thop import profile, clever_format
from nets.MultiScaleAttention import MultiScaleAttention

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train ')
    # parser.add_argument('--pretrained_model',
    #                     default=r'', type=str,
    #                     help='load Pretrained model')
    args = parser.parse_args()
    a = torch.randn(1, 3, 512, 640)
    model = ThermalRGBNet(None)
    flops, params = profile(model, ([a,a],))
    flops, params = clever_format([flops, params], "%.2f")
    print(flops, params)

    c,d = model([a, a])
    # print(b.shape)
    print(c.shape)
    print(d.shape)