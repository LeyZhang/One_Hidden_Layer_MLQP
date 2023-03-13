import random

import torch
import numpy as np
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import Module
import torch.nn as nn


class MLQP(nn.Module):
    def __init__(self, config):
        super(MLQP, self).__init__()
        self.config = config
        self.device = config.device
        self.hidden_num = config.hidden_number
        self.net = nn.Sequential(
            nn.Linear(in_features=4,out_features=self.hidden_num,bias=True),
            nn.Sigmoid(),
            nn.Linear(in_features=self.hidden_num,out_features=1,bias=True),
            nn.Sigmoid())
        print("[INFO] -- MLQP INIT --")

    def forward(self, x):
        x_Q = torch.pow(x,2)
        inputs = torch.cat((x,x_Q),dim=1)
        return self.net(inputs)

class MLQP2(nn.Module):
    def __init__(self, config):
        super(MLQP2, self).__init__()
        self.hidden_num = config.hidden_number
        self.FC1 = nn.Linear(in_features=4,out_features=self.hidden_num,bias=True)
        self.FC2 = nn.Linear(in_features=self.hidden_num*2,out_features=1,bias=True)
        self.sigmoid = nn.Sigmoid()
        print("[INFO] -- MLQP2 INIT --")

    def forward(self, x):
        x_Q = torch.pow(x,2)
        input = torch.cat((x,x_Q),dim=1)
        output = self.FC1(input)
        output = self.sigmoid(output)

        output_Q = torch.pow(output,2)
        output = torch.cat((output,output_Q),dim=1)

        output = self.FC2(output)
        output = self.sigmoid(output)
        return output