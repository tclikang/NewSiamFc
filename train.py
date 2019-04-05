from __future__ import absolute_import, print_function

import os
import sys
import torch
from torch.utils.data import DataLoader

from got10k.datasets import ImageNetVID, GOT10k
from pairwise import Pairwise
from siamfc import TrackerSiamFC
from parameters import param
import utils

if __name__ == '__main__':
    # setup dataset
    para = param()
    name = 'GOT-10k'
    assert name in ['VID', 'GOT-10k']
    if name == 'GOT-10k':
        root_dir = '/home/fanfu/data/GOT-10k'
        seq_dataset = GOT10k(root_dir, subset='train')
    elif name == 'VID':
        root_dir = '/home/fanfu/data/ILSVRC2015/Data/VID/'
        seq_dataset = ImageNetVID(root_dir, subset=('train', 'val'))
    pair_dataset = Pairwise(seq_dataset)

    # setup data loader
    cuda = torch.cuda.is_available()
    loader = DataLoader(
        pair_dataset, batch_size=8, shuffle=True,
        pin_memory=cuda, drop_last=True, num_workers=8)

    # setup tracker
    tracker = TrackerSiamFC()

    # path for saving checkpoints
    net_dir_total = 'pretrained/siamfc_new/'
    if not os.path.exists(net_dir_total):
        os.makedirs(net_dir_total)

    # # 读取训练数据
    # if len(os.listdir(net_dir_feature)) > 0:  # 文件夹不为空
    #     model_list = os.listdir(para.model_save_path)
    #     model_list.sort()
    #     model_path = para.model_save_path + model_list[-1]
    #     tracker.net.load_state_dict(torch.load(model_path))
    #     for name in model_list:
    #         file_name = os.path.join(para.model_save_path, name)
    #         os.remove(file_name)
    #     torch.save(tracker.net.state_dict(), '{}0.pkl'.format(para.model_save_path))

    save_path = net_dir_total
    utils.read_net(net_dir_total, tracker.net)

    epoch_num = 50
    for epoch in range(epoch_num):
        for step, batch in enumerate(loader):
            loss = tracker.step(
                batch, backward=True, update_lr=(step == 0))
            if step % 1 == 0:
                print('Epoch [{}][{}/{}]: Loss: {:.3f}'.format(
                    epoch + 1, step + 1, len(loader), loss))
                sys.stdout.flush()

                # save checkpoint
            if step % 2000 == 0:
                torch.save(tracker.net.state_dict(), '{}E{:0>2d}S{:0>10}.pkl'.format(save_path,epoch,step))

