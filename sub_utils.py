import torch
import datetime
import os
import random
import math

def make_savefolder(result_path):
    """
    Output
    - result_path+Current time foler output
    """
    date_time = datetime.datetime.now()
    date_time_path = date_time.strftime('%y%m%d%H%M')
    result_folder_path = os.path.join(result_path,date_time_path)
    if not os.path.isdir(result_folder_path):
        os.makedirs(result_folder_path, exist_ok=True)
    return result_folder_path

def get_n_params(model):
    """
    get the total number of model parameters
    """
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def adjust_learning_rate(optimizer, epoch, lr, num_epoch):
    """Decay the learning rate based on schedule"""

    # cos lr schedule
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / num_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr