from torch import nn
import torch.nn.functional as F
import torch
import math
import numpy as np
from rep_tdnn import Xvector

'''
The demo of the inference speed test for Rep-TDNN. 
The inference speed margin of used CS-Rep or not will increase with the increasing of  the performance of the device. .
Such as the speed margin of Rep-TDNN with CS-Rep or without CS-Rep on GPU is huger than on CPU.
A huge "inference_samples" will generate more stable results of the inference speed. Such as on a server with 
CPU used 1000 or a server with GPU used 50000 ,etc.
'''


if __name__ == "__main__":
    import torchsummary
    from thop import profile

    million = (100 * 100 * 100)
    ######################################################
    # original topology
    model = Xvector(embedding_size=512)
    model.eval()
    input = torch.randn(1, 161, 300)

    torchsummary.summary(model, (161, 300), device='cpu')

    flops, params = profile(model, inputs=(input,))
    yi = million*1000
    million = 100 * 100 * 100
    print("param:",params / million)
    print("FLOPs:",flops/yi)

    ######################################################
    # cs-rep topology
    model.embedding_net.rep_all()
    input = torch.randn(1, 161, 300)

    torchsummary.summary(model, (161, 300), device='cpu')

    flops, params = profile(model, inputs=(input,))
    yi = million * 1000
    million = 100 * 100 * 100
    print("param:", params / million)
    print("FLOPs:", flops / yi)
