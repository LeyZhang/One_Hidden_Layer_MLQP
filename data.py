import random

import torch
import numpy as np
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import Module
import torch.nn as nn


def generate_data(num):
    r = np.random.rand(num)
    r = r*6
    theta = r * np.pi
    x = np.reshape(r * np.cos(theta), (-1, 1))
    y = np.reshape(r * np.sin(theta), (-1, 1))

    dataset1 = np.concatenate((x, y, np.ones((num, 1))), axis=1)
    dataset0 = np.concatenate((-x, -y, np.zeros((num, 1))), axis=1)
    dataset = np.concatenate((dataset0,dataset1),axis=0)

    np.random.shuffle(dataset)
    dataset = np.float32(dataset)
    return dataset

def get_data():
    #读取数据

    train_data_address = './Dataset/two_spiral_train_data.txt'  # 'Rosslyn_ULA' or 'O1_28B_ULA' or 'I3_60_ULA' or 'O1_28_ULA'
    test_data_address = './Dataset/two_spiral_test_data.txt'

    # train_data = np.loadtxt(train_data_address,dtype=np.float32)
    train_data = generate_data(3000)
    test_data = np.loadtxt(test_data_address,dtype=np.float32)

    return train_data, test_data


def get_loader(train_data, test_data, batch_size):
    train_X,train_Y = torch.from_numpy(train_data[:,0:2]),torch.from_numpy(train_data[:,2])
    test_X,test_Y = torch.from_numpy(test_data[:,0:2]),torch.from_numpy(test_data[:,2])
    #数据封装
    train = torch.utils.data.TensorDataset(train_X,train_Y)
    test = torch.utils.data.TensorDataset(test_X,test_Y)
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)
    return train_loader, test_loader