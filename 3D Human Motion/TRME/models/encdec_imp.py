import torch.nn as nn
import torch.nn.functional as F
from models.resnet_imp import Resnet1D
import math


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.sequential(x)
        return x * scale


class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 dropout=0.1):
        super().__init__()

        self.dropout = dropout
        
        blocks = []
        
        # First layer with normalization and optional dropout
        blocks.append(nn.Conv1d(input_emb_width, width, 3, padding=1))
        blocks.append(nn.BatchNorm1d(width))
        blocks.append(nn.ReLU())
        blocks.append(nn.Dropout(dropout))
        
        # Downsampling layers with SE blocks
        filter_t, pad_t = stride_t * 2, stride_t // 2
        for _ in range(down_t):
            block = nn.Sequential(
                nn.Conv1d(width, width, filter_t, stride_t, pad_t),
                nn.BatchNorm1d(width),
                nn.ReLU(),
                SEBlock(width),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm, dropout=dropout)
            )
            blocks.append(block)

        # Final layer with optional dropout
        blocks.append(nn.Conv1d(width, output_emb_width, 3, padding=1))
        blocks.append(nn.Dropout(dropout))
        
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3, 
                 activation='relu',
                 norm=None,
                 dropout=0.1):
        super().__init__()
        
        self.dropout = dropout
        
        blocks = []
        
        # First layer with normalization
        blocks.append(nn.Conv1d(output_emb_width, width, 3, padding=1))
        blocks.append(nn.BatchNorm1d(width))
        blocks.append(nn.ReLU())
        blocks.append(nn.Dropout(dropout))
        
        # Upsampling layers with residual connections
        for _ in range(down_t):
            block = nn.Sequential(
                SEBlock(width),
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, width, 3, padding=1),
                nn.Dropout(dropout)
            )
            blocks.append(block)
        
        # Final reconstruction layers
        blocks.append(nn.Conv1d(width, input_emb_width, 3, padding=1))
        
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
