import argparse
import numpy as np
import torch

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # self.device = 'cpu'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = True
        self.num_outputs = 1
        self.init_args()
    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)
    
    def init_args(self):
        # self.parser.add_argument('--method', default='mlp', help='[mlp,mlqp]')
        self.parser.add_argument('--seed', default = 7, type = int, help = 'seed')
        self.parser.add_argument('--lr', default = 0.03, type = float, help= 'learning rate')
        self.parser.add_argument('--epochs', default = 50, type = int, help = 'number of iterations')
        self.parser.add_argument('--batch_size', default = 1, type = int, help = 'batch size')
        self.parser.add_argument('--method', default='mlqp', help='[mlp,mlqp,mlqp2]')     
        self.parser.add_argument('--hidden_number', default = 8 ,type = int, help = 'number of hidden node')   
        args = self.parser.parse_args()
        dict_ = vars(args)

        for k, v in dict_.items():
            setattr(self, k, v)
        self.scenario = 'class'

        if self.verbose:
            print("="*80)
            print("[INFO] -- Experiment Configs --")
            print("        training")
            print("           lr:           %5.4f" % self.lr)
            print("         seed:           %d" % self.seed)
            print("       epochs:           %d" % self.epochs)
            # print("   batch size:           %d" % self.batch_size)
            print("       method:           %s" % self.method)
            print("hidden number:           %d" % self.hidden_number)
            print("="*80)
        