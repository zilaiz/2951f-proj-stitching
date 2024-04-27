import torch.nn as nn
from typing import Iterable
import datetime

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_parameters(modules: Iterable[nn.Module]):
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters

## ---------- LUO ------------
import random, torch
import numpy as np
from colorama import Fore

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Custom resolver to get the current time
def get_current_time(fmt="%Y-%m-%d %H:%M:%S"):
    now = datetime.datetime.now()
    return now.strftime(fmt)

def print_color(s, c='r'):
    if c == 'r':
        print(Fore.RED + s + Fore.RESET)
    elif c == 'b':
        print(Fore.BLUE + s + Fore.RESET)
    elif c == 'y':
        print(Fore.YELLOW + s + Fore.RESET)
    else:
        print(Fore.CYAN + s + Fore.RESET)