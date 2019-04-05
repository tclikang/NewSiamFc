from __future__ import absolute_import, division

import numpy as np
from collections import namedtuple
from torch.utils.data import Dataset
from torchvision.transforms import Compose, CenterCrop, RandomCrop, ToTensor
from PIL import Image, ImageStat, ImageOps
from parameters import param
import random

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


class Pairwise(Dataset):

    def __init__(self, seq_dataset, **kargs):
        super(Pairwise, self).__init__()
        self.cfg = self.parse_args(**kargs)
        self.para = param()
        self.seq_dataset = seq_dataset  # 这是一个got10k的dataset类
        self.indices = np.random.permutation(len(seq_dataset))  # 随机打乱seq_dataset的数据，打乱后的索引放到indices
        # augmentation for exemplar and instance images
        self.transform_z = Compose([
            RandomStretch(max_stretch=0.05),  # 这里是对图像进行随机尺度[0.95 1.05)的缩放
            CenterCrop(self.cfg.instance_sz - 8),  # cfg.instance_sz = 255 从图像中间裁剪，尺寸为255-8大小
            RandomCrop(self.cfg.instance_sz - 2 * 8),  # cfg.instance_sz = 255 对图像进行随机裁剪，尺寸为255 - 2*8
            CenterCrop(self.cfg.exemplar_sz),  # cfg.exemplar_sz = 127  然后在中心裁剪成127 * 127
            ToTensor()])
        self.transform_x = Compose([
            RandomStretch(max_stretch=0.05),
            CenterCrop(self.cfg.instance_sz - 8),
            RandomCrop(self.cfg.instance_sz - 2 * 8),
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
        img_files, anno = self.seq_dataset[index]  # 得到第index中的所有文件名,和对应的包围框
        # 获得训练样本,包括一个连续十四帧的sequence
        # img_files[rand_z_seq]表示需要用来训练的样本数量
        rand_z_seq, rand_x = self.lk_sample_seq_pair(len(img_files), self.para.prior_frames_num)

        # rand_z, rand_x = self._sample_pair(len(img_files))  # 从第index的序列中选取两帧id，其中rand_z是z的帧号

        exemplar_images = []
        for i in range(len(rand_z_seq)):
            temp = Image.open(img_files[rand_z_seq[i]])  # 读取z的图像
            temp = self._crop_and_resize(temp, anno[rand_z_seq[i]])  # 这里的裁剪都是255大小的,经过transform_z之后变成127
            temp = 255.0 * self.transform_z(temp)
            exemplar_images.append(temp)
        instance_image  = Image.open(img_files[rand_x])  # 读取x的图像
        # exemplar_image  = self._crop_and_resize(exemplar_image, anno[rand_z])
        instance_image = self._crop_and_resize(instance_image, anno[rand_x])
        # exemplar_image  = 255.0 * self.transform_z(exemplar_image)  # 这里永远是127大小的图片
        instance_image_z_target = 255.0 * self.transform_z(instance_image)  # 用于特征的loss
        instance_image_search = 255.0 * self.transform_x(instance_image)
        # print(instance_image.size())
        # exemplar_images 是prior_frames_num个数的训练图片序列,list格式,每个里面保存一张裁剪后的图片
        # instance_image_z_target 用来比对特征的图片
        # instance_image_search 用来search的图片
        return exemplar_images, instance_image_z_target, instance_image_search

    # len = 93350
    # self.cfg.pairs_per_seq = 10
    # self.seq_dataset = 9335
    def __len__(self):
        return self.cfg.pairs_per_seq * len(self.seq_dataset)


    # vid_len表示视频的长度, prior_sample_len表示需要用多少样本去预测下一帧的值
    def lk_sample_seq_pair(self, vid_len, prior_samples_len):
        assert vid_len > (prior_samples_len + 1)
        start_frame = random.randint(0, vid_len - (prior_samples_len + 1) - 1)  # 下标从0开始的那种
        rand_z_seq = list(range(start_frame, start_frame + prior_samples_len))
        rand_x = start_frame + prior_samples_len + 1
        return rand_z_seq, rand_x

    # 一个视频总共有n帧，编号是从0 到 n-1
    def _sample_pair(self, n):
        assert n > 0
        if n == 1:
            return 0, 0
        elif n == 2:
            return 0, 1
        else:
            # self.cfg.max_dist = 100 这说明，最大的跨度，也就是论文中的T最大值为100
            max_dist = min(n - 1, self.cfg.max_dist)
            rand_dist = np.random.choice(max_dist) + 1  # 从0到max_dist中选择一个，选择一个跨度T
            rand_z = np.random.choice(n - rand_dist)  # z从0到n-T之间选一个
            rand_x = rand_z + rand_dist  # x 等于 z的帧号+跨度

        return rand_z, rand_x


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
