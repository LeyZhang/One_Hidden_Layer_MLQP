import math
import numpy as np
import time

import scipy.io as sio
import torch.utils.data
import torch.optim as optim
import torch.nn as nn

from data import *
from plot import *
from model import *
from config import Config
from utils import *



        
def run(config):
    # parameter
    lr = config.lr
    batch_size = config.batch_size
    epochs = config.epochs
    np.random.seed(config.seed)
    device =  torch.device(config.device)

    # address
    plt_save_address =  f"./model={config.method}_lr={config.lr}_seed={config.seed}_hidden_num={config.hidden_number}"
    # if os.path.exists(plt_save_address):
    #     print("[INFO]: The experiment Pass")
    #     return

    print("[INFO]: the plt save address is " + plt_save_address)
    check_path(plt_save_address)
    print(device)
    early_stopping = EarlyStopping(plt_save_address)
    # data
    train_data, test_data = get_data()
    train_loader, test_loader = get_loader(train_data, test_data, batch_size)




    
    fn_map = {
        "mlp":  MLP,
        "mlqp": MLQP,
        "mlqp2": MLQP2
    }
    #model training
    model = fn_map[config.method](config).to(device)
    # opt = optim.Adam(model.parameters(),lr=lr)
    opt = optim.SGD(model.parameters(), lr=lr)

    st = time.time()
    train_loss_hist, train_acc_hist,test_loss_hist,test_acc_hist = fit_model(model=model, train_loader=train_loader,test_loader=test_loader,opt=opt,
                                                                    loss_fn=nn.MSELoss(),epochs=epochs, device = device,early_stopping  = early_stopping )
    st = time.time() - st
    print('time cost: {time} sec'.format(time=st))
    
    get_plot(train_acc_hist, train_loss_hist, test_acc_hist, test_loss_hist, lr, plt_save_address, model, device)
    visualize_boundary(model, plt_save_address, device, test_data)
   


if __name__ == "__main__":

    config = Config()

    run(config)