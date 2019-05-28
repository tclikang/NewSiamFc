from __future__ import absolute_import, division
import numpy as np
from collections import namedtuple
from torch.utils.data import Dataset
from torchvision.transforms import Compose, CenterCrop, RandomCrop, ToTensor
from PIL import Image, ImageStat, ImageOps
import random
import torch
from torch.utils.data import DataLoader
from got10k.datasets import ImageNetVID, GOT10k
from parameters import param
import test_bb_reg.bb_tools as bb_tools
import visdom
import test_bb_reg.utils as tu
import torchvision.transforms as TT
import torchvision.transforms.functional as F


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class RandomStretch(object):

    def __init__(self, max_stretch=0.05, interpolation='bilinear'):
        assert interpolation in ['bilinear', 'bicubic']
        self.max_stretch = max_stretch
        self.interpolation = interpolation

    # 这个函数就是对图像进行一个随机的尺度从[0.95 1.05)比例的缩放，缩放使用的是二次插值
    def __call__(self, img):
        # np.random.uniform是从[low high)的均匀分布中采样，这里就是从[-0.05 0.05)之间的缩放
        scale = 1.0 + np.random.uniform(
            -self.max_stretch, self.max_stretch)
        # img.size = (255 255),下面是先对255大小的图像进行一个缩放，然后取整转换成一个int大小的尺度
        # size是缩放后的尺度
        size = np.round(np.array(img.size, float) * scale).astype(int)
        if self.interpolation == 'bilinear':
            method = Image.BILINEAR
        elif self.interpolation == 'bicubic':
            method = Image.BICUBIC
        # 下面就是对图像进行插值缩放
        return img.resize(tuple(size), method)


class Mydataset(Dataset):

    def __init__(self, seq_dataset, **kargs):
        super(Mydataset, self).__init__()
        self.cfg = self.parse_args(**kargs)
        self.para = param()
        self.seq_dataset = seq_dataset  # 这是一个got10k的dataset类
        self.indices = np.random.permutation(len(seq_dataset))  # 随机打乱seq_dataset的数据，打乱后的索引放到indices
        # augmentation for exemplar and instance images
        self.transform_z = Compose([
            CenterCrop(self.cfg.exemplar_sz),  # cfg.exemplar_sz = 127  然后在中心裁剪成127 * 127
            ToTensor()])
        self.transform_x = Compose([
            CenterCrop(self.cfg.exemplar_sz),  # cfg.exemplar_sz = 127  然后在中心裁剪成127 * 127
            ToTensor()])

    def parse_args(self, **kargs):
        # default parameters
        cfg = {
            'pairs_per_seq': 8,  # 一个seq里面弄多少pairs, default 10
            'max_dist': 100,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
        }

        for key, val in kargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('GenericDict', cfg.keys())(**cfg)

    # 返回一个pairs，其中z的大小是127，x的大小是239
    def __getitem__(self, index):
        # len(self.seq_dataset) = 9335
        # index = 5218 这是首先选择一个数据集的编号
        index = self.indices[index % len(self.seq_dataset)]
        # 把这个数据集中所有图像的路径保存在img_files中，包围框保存在anno中
        # 这里的anno是xywh topleft类型的
        img_files, anno = self.seq_dataset[index]  # 得到第index中的所有文件名,和对应的包围框

        # 获得训练样本,包括一个连续十四帧的sequence
        # img_files[rand_z_seq]表示需要用来训练的样本数量
        z_index, x_index = self.sample_training_data_index(len(img_files))
        z_img = Image.open(img_files[z_index])  # 读取z
        x_img = Image.open(img_files[x_index])  # 读取x图像
        # 对x的anno做一个小小的扰动,然后再进行裁剪
        z_anno = anno[z_index]  # z图像的包围框
        x_anno = anno[x_index]  # x图像的包围框
        w = x_img.width
        h = x_img.height
        x_anno_minmax = bb_tools.bb_from_xywh_topleft_to_minmax(x_anno)  # 转化成minmax
        x_anno_rao = bb_tools.raodong_bb(x_anno_minmax, w, h, 0.05, 0.05)  # 进行扰动
        x_anno_rao = bb_tools.bb_from_minmax_to_xywh_topleft(x_anno_rao)  # 转成topleft
        # -----------裁剪x,z,将它们放入tensor
        x_img_crop = self._crop_and_resize(x_img, x_anno_rao)
        z_img_crop = self._crop_and_resize(z_img, z_anno)
        x_img_crop = self.transform_x(x_img_crop) * 255.0
        z_img_crop = self.transform_z(z_img_crop) * 255.0
        # ------------求出regression 分数
        reg = tu.get_tx_ty_tw_th_from_two_anno(x_anno, x_anno_rao)
        # ------------将anno格式转成xmin格式
        z_loc = bb_tools.bb_from_xywh_topleft_to_minmax(z_anno)  # 全部转成xmin格式
        x_loc = bb_tools.bb_from_xywh_topleft_to_minmax(x_anno)  # 转成xmin格式
        x_loc_rao = bb_tools.bb_from_xywh_topleft_to_minmax(x_anno_rao)  # 转乘xmin格式
        # -----------需要显示的图片
        # z_img_show = F.resize(z_img, (300, 300))
        # x_img_show = F.resize(x_img, (300, 300))
        # z_img_show = TT.ToTensor()(z_img_show) * 255.0
        # x_img_show = TT.ToTensor()(x_img_show) * 255.0
        return img_files[z_index], img_files[x_index], z_img_crop, x_img_crop, z_loc.astype(float), \
               x_loc.astype(float), x_loc_rao.astype(float), reg.astype(float)


    # len = 93350
    # self.cfg.pairs_per_seq = 10
    # self.seq_dataset = 9335
    def __len__(self):
        return self.cfg.pairs_per_seq * len(self.seq_dataset)


    def sample_training_data_index(self, vid_len):
        assert vid_len > 2
        start_frame = random.randint(0, vid_len-2)
        rand_x = random.randint(start_frame + 1, vid_len-1)
        return start_frame, rand_x



    # image是PIL图像，box是目标的xywh
    # 这里的box中的坐标x是从1开始记录的，下面要从0的索引开始
    def _crop_and_resize(self, image, box):
        # convert box to 0-indexed and center based
        box = np.array([
            box[0] - 1 + (box[2] - 1) / 2,
            box[1] - 1 + (box[3] - 1) / 2,
            box[2], box[3]], dtype=np.float32)
        center, target_sz = box[:2], box[2:]  # center是目标中心索引，从0开始 target_sz是尺度

        # exemplar and search sizes
        # self.cfg.context = 0.5
        # np.sum(target_sz)是将目标的长宽相加
        context = self.cfg.context * np.sum(target_sz)  # (w+h)*0.5
        # 这里z_sz乘以scale之后才是127的大小
        # 这里x_sz乘以scale之后才是255的大小
        z_sz = np.sqrt(np.prod(target_sz + context))  # eq.7     s*z_sz = 127
        x_sz = z_sz * self.cfg.instance_sz / self.cfg.exemplar_sz  # s*x_sz = 255

        # convert box to corners (0-indexed)
        size = round(x_sz)
        # 以center为中心两边同时减去和加上半径
        corners = np.concatenate((
            np.round(center - (size - 1) / 2),
            np.round(center - (size - 1) / 2) + size))
        # [398,  426, 1197, 1225] 表示
        corners = np.round(corners).astype(int)

        # pad image if necessary
        # 需要pad的尺寸大小
        pads = np.concatenate((
            -corners[:2], corners[2:] - image.size))
        npad = max(0, int(pads.max()))
        if npad > 0:
            # 如果需要padding，下面是3层的padding的值
            avg_color = ImageStat.Stat(image).mean
            # PIL doesn't support float RGB image
            avg_color = tuple(int(round(c)) for c in avg_color)
            # 进行papdding
            image = ImageOps.expand(image, border=npad, fill=avg_color)

        # crop image patch
        # padding后的坐标应该加上padding的值
        corners = tuple((corners + npad).astype(int))
        # 进行裁剪
        patch = image.crop(corners)

        # resize to instance_sz
        out_size = (self.cfg.instance_sz, self.cfg.instance_sz)
        patch = patch.resize(out_size, Image.BILINEAR)
        # 最后返回的的是一个255大小的图片，裁剪后的255大小的图片
        return patch


if __name__ == '__main__':
    vis = visdom.Visdom()
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
    pair_dataset = Mydataset(seq_dataset)

    # setup data loader
    cuda = torch.cuda.is_available()
    loader = DataLoader(
        pair_dataset, batch_size=1, shuffle=True,
        pin_memory=cuda, drop_last=True, num_workers=1)

    for step, (a,b,z_img, x_img, z_anno, x_anno, x_anno_rao,reg) in enumerate(loader):
        vis.close()  # 关闭显示
        vis.images(z_img, opts=dict(title='z_img', caption='How random.'))
        vis.images(x_img, opts=dict(title='x_img', caption='How random.'))
        pass
