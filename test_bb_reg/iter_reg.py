import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import test_bb_reg.utils as test_utils
import visdom
from test_bb_reg.dataset import *
import test_bb_reg.bb_tools as tb
import test_bb_reg.utils as tu
import torch.optim as optim
import test_bb_reg.visualize_tools as tv
import cv2
import test_bb_reg.file_tools as file_tools
from torch.optim.lr_scheduler import ExponentialLR
import test_bb_reg.net_work as net_work
from PIL import Image, ImageStat, ImageOps
import time

if __name__ == '__main__':
    # 初始化可视化工具
    vis = visdom.Visdom()
    model_file = '/home/fanfu/PycharmProjects/SimpleSiamFC/test_bb_reg/model_temp/'
    # 初始化模型
    model = net_work.MyNet()
    model = model.to(model.device)
    # 预读取模型
    current_epoch = file_tools.test_read_net(model_file, model)
    # setup dataset
    name = 'GOT-10k'
    assert name in ['VID', 'GOT-10k']
    if name == 'GOT-10k':
        root_dir = '/home/fanfu/data/GOT-10k'
        seq_dataset = GOT10k(root_dir, subset='train')
    elif name == 'VID':
        root_dir = '/home/fanfu/data/ILSVRC2015/Data/VID/'
        seq_dataset = ImageNetVID(root_dir, subset=('train', 'val'))

    # 初始化数据load工具
    pair_dataset = Mydataset(seq_dataset)

    # setup data loader
    cuda = torch.cuda.is_available()
    loader = DataLoader(
        pair_dataset, batch_size=1, shuffle=True, pin_memory=cuda,
        drop_last=True, num_workers=1)

    show_loss = 0
    for step, (z_img_files, x_img_files, z_img, x_img,
               z_loc, x_loc, x_loc_rao, reg_score) in enumerate(loader):
        with torch.no_grad():
            # 将需要计算的图片放入cuda中
            z_img = Image.open(z_img_files[0])
            x_img = Image.open(x_img_files[0])
            z_loc = z_loc.numpy()[0]
            x_loc_rao = x_loc_rao.numpy()[0]
            x_new_loc = x_loc_rao
            # 第12个epoch的效果是很好的
            for i in range(3):
                x_new_loc = test_utils.do_bbreg_according_to_network(model, z_img, z_loc, x_img, x_new_loc)
            # ------------------相关作图代码-------------------------------------

            vis.close()  # 关闭显示
            z_img_show = F.to_tensor(tv.read_img(z_img_files[0])).unsqueeze(0).numpy() * 255.0
            x_img_show = F.to_tensor(tv.read_img(x_img_files[0])).unsqueeze(0).numpy() * 255.0
            z_img_show = tv.show_torch_img_with_bblist(z_img_show, [z_loc])
            x_img_show1 = tv.show_torch_img_with_bblist(x_img_show, [x_loc_rao])
            x_img_show2 = tv.show_torch_img_with_bblist(x_img_show, [x_new_loc])
            vis.images(np.vstack((z_img_show, x_img_show1, x_img_show2)),
                       opts=dict(title='test ', caption='----------------------'))
            time.sleep(5)
