from __future__ import absolute_import

from got10k.experiments import *

from siamfc import TrackerSiamFC
import os
from parameters import param
import torch

if __name__ == '__main__':
    # setup tracker
    # 测试合并分值
    net_path = '/home/fanfu/PycharmProjects/SimpleSiamFC/pretrained/siamfc_new'
    tracker = TrackerSiamFC(net_path=net_path)
    # load net
    para = param()
    if len(os.listdir(para.model_save_path)) > 0:  # 文件夹不为空
        model_list = os.listdir(para.model_save_path)
        model_list.sort()
        model_path = para.model_save_path + model_list[-1]
        tracker.net.load_state_dict(torch.load(model_path))
        # for name in model_list:
        #     file_name = os.path.join(para.model_save_path, name)
        #     os.remove(file_name)
        # torch.save(tracker.net.state_dict(), '{}0.pkl'.format(para.model_save_path))


    # setup experiments
    experiments = [
        # ExperimentGOT10k('/home/fanfu/data/GOT-10k', subset='test'),
        ExperimentOTB('/home/fanfu/data/OTB', version=2013),
        # ExperimentOTB('/home/fanfu/data/OTB', version=2015),
        # ExperimentVOT('data/vot2018', version=2018),
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
