#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:16:35 2019

@author: fanghenshao
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class netGv(nn.Module):
    def __init__(self, output_channels, nz=100, nfm=32):
        super(netGv, self).__init__()
        self.output_channels = output_channels

        # generate kernel with size = 5
        self.generator = nn.Sequential(
                # ---------------------------------------------------------
                nn.ConvTranspose2d(in_channels=nz, out_channels=nfm*2,
                                    kernel_size=2, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(nfm*2),
                nn.ReLU(True),
                # state size. (nfm*2) x 2 x 2
                
                # ---------------------------------------------------------
                nn.ConvTranspose2d(in_channels=nfm*2, out_channels=nfm,
                                    kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(nfm),
                nn.ReLU(True),
                # state size. (nfm) x 3 x 3
                
                # ---------------------------------------------------------
                nn.ConvTranspose2d(in_channels=nfm, out_channels=nfm,
                                    kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(nfm),
                nn.ReLU(True),
                # state size. (nfm) x 5 x 5
                
                # ---------------------------------------------------------
                nn.ConvTranspose2d(in_channels=nfm, out_channels=output_channels,
                                    kernel_size=3, stride=1, padding=1, bias=False),
                nn.Tanh()
                # state size. (D) x 5 x 5
                )
    def forward(self, noise):
        return self.generator(noise)
                    


class GRFFNetv(nn.Module):
    """
    SIMULATE THE STRUCTURE OF LENET-5:
        1. Adopting a generator to generate 2-D kernels
        2. Keeping the convolution-relu-maxpooling2d manipulations
        3. Adopting a generator to generate weights
        4. Replacing the MLP part in LeNet-5 with Multi-layer GRFF
    """
    def __init__(self, GRFFBlock):
        super(GRFFNetv, self).__init__()
        self.num_layers = 3
        self.main = nn.Sequential(*[GRFFBlock(output_channels=1, nz=100, nfm=32),
                                    GRFFBlock(output_channels=32, nz=100, nfm=32),
                                    nn.Linear(16*4*4, 10)])
        self.kernel1 = 0
        self.kernel2 = 0
    def forward(self, x, noise):

        kernels1 = self.main[0].forward(noise[0])
        # -------- simulate the RFF mapping function
        x_1 = F.max_pool2d(torch.cos(F.conv2d(x, kernels1, stride=1, padding=0)), (2,2)) # 1x28x28 -> 16x24x24 -> 16x12x12
        x_2 = F.max_pool2d(torch.sin(F.conv2d(x, kernels1, stride=1, padding=0)), (2,2)) # 1x28x28 -> 16x24x24 -> 16x12x12
        x = torch.cat((x_1, x_2),1) # 32x12x12

        kernels2 = self.main[1].forward(noise[1])
        # -------- simulate the RFF mapping function
        x_1 = F.max_pool2d(torch.cos(F.conv2d(x, kernels2, stride=1, padding=0)), (2,2)) # 32x12x12 -> 8x8x8 -> 8x4x4
        x_2 = F.max_pool2d(torch.sin(F.conv2d(x, kernels2, stride=1, padding=0)), (2,2)) # 32x12x12 -> 8x8x8 -> 8x4x4
        x = torch.cat((x_1, x_2),1) # 16x4x4
        
        x=x.view(x.size(0), -1)
        out = self.main[2](x)

        return out, kernels1, kernels2
        
        for idx in range(self.num_layers):
            # ------------------- forward, conv. layer-1
            if idx==0:
                conv_layer = self.main[idx]
                kernels = conv_layer.forward(noise[idx])
                # -------- simulate the RFF mapping function
                x_1 = F.max_pool2d(torch.cos(F.conv2d(x, kernels, stride=1, padding=0)), (2,2)) # 1x28x28 -> 16x24x24 -> 16x12x12
                x_2 = F.max_pool2d(torch.sin(F.conv2d(x, kernels, stride=1, padding=0)), (2,2)) # 1x28x28 -> 16x24x24 -> 16x12x12
                x = torch.cat((x_1, x_2),1) # 32x12x12
                self.kernel1 = kernels
            # ----------------- forward, conv. layer-2
            if idx==1:
                conv_layer = self.main[idx]
                kernels = conv_layer.forward(noise[idx])
                # -------- simulate the RFF mapping function
                x_1 = F.max_pool2d(torch.cos(F.conv2d(x, kernels, stride=1, padding=0)), (2,2)) # 32x12x12 -> 8x8x8 -> 8x4x4
                x_2 = F.max_pool2d(torch.sin(F.conv2d(x, kernels, stride=1, padding=0)), (2,2)) # 32x12x12 -> 8x8x8 -> 8x4x4
                x = torch.cat((x_1, x_2),1) # 16x4x4
                self.kernel2 = kernels
                x=x.view(x.size(0), -1)
            # ----------------- forward, fc layer
            if idx==2:
                fc_layer = self.main[idx]
                out = fc_layer(x)
        return out, self.kernel1, self.kernel2








