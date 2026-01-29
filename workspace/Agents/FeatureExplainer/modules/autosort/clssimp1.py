import numpy as np
import torch
import torch
import torch.nn as nn
import numpy as np

class clssimp(nn.Module):
    def __init__(self, ch=2880, num_classes=20, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.pool = nn.AdaptiveAvgPool1d(output_size=(ch))
        self.way1 = nn.Sequential(
            nn.Linear(ch, 1024, bias=True),#1024 for npix 1000 for mesh
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        self.way2 = nn.Sequential(
            nn.Linear(1024, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.way3 = nn.Sequential(
            nn.Linear(512, 256, bias=True),#256 fir npix 100 for mesh
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Linear(256, num_classes, bias=True)
        
        self.input_size = ch

    def forward(self, x):
        x = x[:,:self.input_size].float()
        x = self.pool(x[None, :])
        x = x.reshape(x.size(1), -1)
        x = x.to(torch.float32)
        x = self.way1(x)
        x = self.way2(x)
        x = self.way3(x)
        logits = self.cls(x)
        return logits

    def intermediate_forward(self, x):
        x = self.pool(x[None, :])
        x = x.reshape(x.size(1), -1)
        x = self.way1(x)
        x = self.way2(x)
        x = self.way3(x)
        return x