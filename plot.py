import glob
import matplotlib.pyplot as plt
import torch
from config import Config
import torch.nn.functional as F
import numpy as np
from scipy import stats as st
from numpy import unique,where

def get_plot(train_acc_hist, train_loss_hist, test_acc_hist, test_loss_hist, lr, plt_save_address, model, device):
    plt.figure()
    plt.plot(train_acc_hist, label='training acc', color = '#8ECFC9')
    plt.plot(test_acc_hist, label='validation acc', color = '#FFBE7A')
    plt.legend()
    plt.title('Acc hist,lr = {}'.format(lr))
    plt.savefig(plt_save_address + '/Acc hist.png')
    plt.show()

    plt.figure()
    plt.plot(train_loss_hist, label='training loss',color = '#8ECFC9')
    plt.plot(test_loss_hist, label='validation loss', color = '#FFBE7A')
    plt.legend()
    plt.title('Loss hist,lr = {}'.format(lr))
    plt.savefig(plt_save_address + '/Loss hist.png')
    plt.show()

def plot_data(test_data):
    X , y = test_data[:,0:2] , test_data[:,2]
    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0

    # Plot Examples
    plt.plot(X[pos, 0], X[pos, 1], 'X', mew=1, ms=10, mec='k')
    plt.plot(X[neg, 0], X[neg, 1], 'o', mew=1, mfc='y', ms=10, mec='k')

def visualize_boundary(model, plt_save_address, device, test_data):
    plt.figure()
    plot_data(test_data)
    blank = torch.zeros((int(12/0.05),int(12/0.05),1))

    x = torch.arange(-6,6,0.05)
    x = torch.reshape(x,(-1,1,1))
    x = blank + x

    y = torch.arange(-6,6,0.05)
    y = torch.reshape(y,(1,-1,1))
    y = blank + y

    axis = torch.cat((x,y),dim = 2).to(device)
    axis = torch.reshape(axis,(240*240,2))
    classify = (model(axis).cpu()>=0.5)
    
    # Negative Point
    row_ix = where(classify == False)
    x = axis[row_ix, 0].cpu()
    y = axis[row_ix, 1].cpu()
    plt.scatter(x,y, c = '#999999')

    # Positive Point
    row_ix_1 = where(classify == True)
    x_1 = axis[row_ix_1, 0].cpu()
    y_1 = axis[row_ix_1, 1].cpu()
    plt.scatter(x_1,y_1, c = '#82B0D2')


    plt.savefig(plt_save_address + '/Classify.png')
    plt.show()