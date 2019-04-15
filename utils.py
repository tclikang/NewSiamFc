from __future__ import absolute_import, print_function

import os
import sys
import torch
from torch.utils.data import DataLoader

from got10k.datasets import ImageNetVID, GOT10k
from pairwise import Pairwise
from siamfc import TrackerSiamFC
from parameters import param

# 讲最新的模型读入net,并删除老模型
def read_net(filepath, net, optimizer):
    # 读取训练数据
    if len(os.listdir(filepath)) > 0:  # 文件夹不为空
        model_list = os.listdir(filepath)
        model_list.sort()
        model_path = filepath + model_list[-1]
        loaded_model = torch.load(model_path)
        net.load_state_dict(loaded_model['state_dict'])
        optimizer.load_state_dict(loaded_model['optimizer'])
        for name in model_list:
            file_name = os.path.join(filepath, name)
            os.remove(file_name)
        save_model(0, net, optimizer, filepath, 0)
        return loaded_model['epoch']
    else:
        return 0
        # torch.save(net.state_dict(), '{}0.pkl'.format(filepath))

def save_model(epoch, net, optimizer, save_path, step):
    torch.save({'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()},
               '{}E{:0>2d}S{:0>10}.pkl'.format(save_path, epoch, step))







