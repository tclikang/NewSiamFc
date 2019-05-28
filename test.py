from __future__ import absolute_import

from got10k.experiments import *

from siamfc import TrackerSiamFC
import os
from parameters import param
import torch
from train import load_pre_train_mode
import utils
import test_bb_reg.net_work
import test_bb_reg.file_tools as tbr_f
import test_bb_reg.utils as tbr_u

if __name__ == '__main__':
    # setup tracker
    # net_path = '/home/fanfu/PycharmProjects/SimpleSiamFC/pretrained/siamfc_new'
    net_pretrain_path = '/home/fanfu/PycharmProjects/SimpleSiamFC/pretrained/good_parameters/'
    para = param()
    tracker = TrackerSiamFC()


    # load net
    # load_pre_train_mode(tracker.net, net_pretrain_path)  # load pretrain model
    # loading my train model
    if len(os.listdir(net_pretrain_path)) > 0:  # 文件夹不为空
        model_list = os.listdir(net_pretrain_path)
        model_list.sort()
        model_path = net_pretrain_path + model_list[-1]
        tracker.net.load_state_dict(torch.load(model_path))


    # setup experiments
    experiments = [
        #ExperimentGOT10k('/home/fanfu/data/GOT-10k', subset='test'),
        ExperimentOTB('/home/fanfu/data/OTB', version=2013),
        # ExperimentOTB('/home/fanfu/data/OTB', version=2015),
        # ExperimentVOT('/home/fanfu/data/vot2018', version=2018),
        # ExperimentDTB70('data/DTB70'),
        # ExperimentTColor128('data/Temple-color-128'),
        # ExperimentUAV123('data/UAV123', version='UAV123'),
        # ExperimentUAV123('data/UAV123', version='UAV20L'),
        # ExperimentNfS('data/nfs', fps=30),
        # ExperimentNfS('data/nfs', fps=240)
    ]

    # run tracking experiments and report performance
    for e in experiments:
        e.run(tracker, visualize=True)
        e.report([tracker.name])
