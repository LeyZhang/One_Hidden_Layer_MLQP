import random

import torch
import numpy as np
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import Module
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, config) :
        super(MLP, self).__init__()

        self.config = config
        self.device = config.device
        self.hidden_num = config.hidden_number
        self.hidden = nn.Linear(in_features=2,out_features=self.hidden_num,bias=True)
        self.output = nn.Linear(in_features=self.hidden_num , out_features=1,bias=True)

        self.net = nn.Sequential(
            nn.Linear(in_features=2,out_features=self.hidden_num,bias=True),
            nn.Sigmoid(),
            nn.Linear(in_features=self.hidden_num,out_features=1,bias=True),
            nn.Sigmoid())

        self.sigmoid = nn.Sigmoid()
        
        print("[INFO] -- MLP INIT --")


    def forward(self, x):
        inputs = x
        return self.net(inputs)