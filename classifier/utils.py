import torch
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import datasets 
from sklearn.manifold import TSNE 
from sklearn.cluster import KMeans 
import pdb

def cosine_schedule_warmup(total_step, value, final_value=0, warmup_step=0, warmup_value=0):
    if warmup_step > 0:
        warmup_schedule = np.linspace(warmup_value, value, warmup_step+2)[1:-1]
    else:
        warmup_schedule = np.array([])
    steps = np.arange(total_step - warmup_step)
    schedule = final_value + 0.5 * (value-final_value) * (1+np.cos(np.pi * steps / len(steps)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == total_step
    return schedule

class build_cosine_scheduler:
    def __init__(self, optimizer, lr, total_step, lr_warmup_step=0):
        init_lr = 0
        final_lr = lr * 1e-3
        self.lrs = cosine_schedule_warmup(total_step, lr, final_lr, lr_warmup_step, init_lr)
        self.optimizer = optimizer

    def step(self,idx):
        lr = self.lrs[idx]
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"]= lr
        self.lr=lr

class build_bicosine_scheduler:
    def __init__(self, optimizer, lr, total_step, lr_warmup_step=0):
        lr_promt = lr[0]
        lr_conv = lr[1]
        init_lr=0
        final_lr_promt = lr_promt * 1e-3
        final_lr_conv = lr_conv * 1e-3
        self.lrs_prompt = cosine_schedule_warmup(total_step, lr_promt, final_lr_promt, lr_warmup_step, init_lr)
        self.lrs_conv = cosine_schedule_warmup(total_step, lr_conv, final_lr_conv, lr_warmup_step, init_lr)
        self.optimizer = optimizer

    def step(self,idx):
        lr_promt = self.lrs_prompt[idx]
        lr_conv = self.lrs_conv[idx]
        for i, param_group in enumerate(self.optimizer.param_groups):
            # pdb.set_trace()
            if i==0:
                param_group["lr"] = lr_conv
            else:
                param_group["lr"] = lr_promt 
        self.lr_conv = lr_conv
        self.lr_prompt = lr_promt


def cosine_loss(q,k):           #q:(32,1,768) k:(32,3,768)
    q = q.repeat(1,k.shape[1],1)#q:(32,1,768)->(32,3,768)
    k = k/k.norm(dim=-1,keepdim=True)
    cos = ((q*k)/(k.shape[0]*k.shape[1])).sum() #相似度
    return 1-cos    #将相似度转换为loss

def cosine_loss_cp(q,k):#q:(32,1,768) k:(64,3,768)
    q = q/q.norm(dim=-1,keepdim=True)
    k = k/k.norm(dim=-1,keepdim=True)
    cos = ((q*k)/(k.shape[0]*k.shape[1])).sum()
    return 1-cos
