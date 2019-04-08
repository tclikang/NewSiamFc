from __future__ import absolute_import, division

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from imageprocess import showimg,showbb
from graphviz import Digraph
from torch.autograd import Variable
from make_dot import make_dot

from got10k.trackers import Tracker
import numpy as np
from imageprocess import showbb
from parameters import param

class SiamFC(nn.Module):

    def __init__(self):
        super(SiamFC, self).__init__()
        # 这里讲一下conv2中的group参数
        # 假设当前的输入channel数为8， 输出为12，卷积核大小为3，
        # 则当group=1时， 总共有12个神经元，每个神经元的参数大小为3*3*8，每个神经元生成一个channel
        # 假设当group=4的时候，每一组的神经元输入channel=2，输出=3，总共仍然有12个神经元，只不过每个神经元的大小为3*3*2，而不是3*3*8
        # Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), groups=2)
        # self.feature[4].weight.data.shape = torch.Size([256, 48, 5, 5])
        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        self.para = param()
        self.feature = nn.Sequential(
            # conv1
            nn.Conv2d(3, 96, 11, 2),  # 这里的权值是[96, 3, 11, 11]
            nn.BatchNorm2d(96, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv2
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv3
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(384, 256, 3, 1, groups=2))
        self._initialize_weights()
        self.deconv = nn.Conv2d(256 * self.para.prior_frames_num, 256, 1, 1)  # 用于将堆叠后的特征对齐


    # seq_z是保存这13个用于帧序列的图像,是一个list,每个
    # 已经重新验证,无误
    def lk_forward(self, z_seq, x_target, x_search):
        z_feat_cat = None
        for z in z_seq:
            z_feat = self.feature(z.to(self.device))
            if z_feat_cat is None:
                z_feat_cat = z_feat
            else:
                z_feat_cat = torch.cat((z_feat_cat,z_feat),1)

        z_feat_deconv = self.deconv(z_feat_cat)  # 6 6 256
        x_target_feat = self.feature(x_target.to(self.device))  # 6  6  256
        x_search_feat = self.feature(x_search.to(self.device))  # 22 22 256

        # fast cross correlation
        n, c, h, w = x_search_feat.size()
        x = x_search_feat.view(1, n * c, h, w)
        # conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        # 这里的第二个参数weight的是out_channel in_channel kernel_size kernel_size
        # 这里的input的input_channel out_channel img_size_h img_size_w
        # 当group=n时，input按照out_channel划分，output按照in_channel划分
        out = F.conv2d(x, z_feat_deconv, groups=n)
        out = out.view(n, 1, out.size(-2), out.size(-1))

        # adjust the scale of responses
        response17 = 0.001 * out + 0.0
        # z_feat_deconv 是127的特征图,6*6*256大小的
        # x_target_feat 是127特征图,是14帧的目标区域
        # response17是卷积后的特征图
        return z_feat_deconv, x_target_feat, response17

    def forward(self, z_seq, x_target, x_search):
        # z = self.feature(z)  # z 的尺寸是 8 256 6 6 这里确实是256 而不是像论文中的128
        # x = self.feature(x)  # x 的尺寸是 8 256 20 20  这里确实是256 而不是像论文中的128
        #
        # # fast cross correlation
        # n, c, h, w = x.size()
        # x = x.view(1, n * c, h, w)
        # # conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        # # 这里的第二个参数weight的是out_channel in_channel kernel_size kernel_size
        # # 这里的input的input_channel out_channel img_size_h img_size_w
        # # 当group=n时，input按照out_channel划分，output按照in_channel划分
        # out = F.conv2d(x, z, groups=n)
        # out = out.view(n, 1, out.size(-2), out.size(-1))
        #
        # # adjust the scale of responses
        # out = 0.001 * out + 0.0
        #
        # return out
        return self.lk_forward(z_seq, x_target, x_search)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, mode='fan_out',
                                     nonlinearity='relu')
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None,**kargs):
        super(TrackerSiamFC, self).__init__(
            name='SiamFC', is_deterministic=True)
        self.cfg = self.parse_args(**kargs)
        self.para = param()
        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = SiamFC()
        # if net_path is not None:
        #     self.net.load_state_dict(torch.load(
        #     net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.deconv.parameters(),
            lr=self.cfg.initial_lr,  # self.cfg.initial_lr = 0.01
            weight_decay=self.cfg.weight_decay,  # self.cfg.weight_decay = 0.0005
            momentum=self.cfg.momentum)  # self.cfg.momentum = 0.9


        # setup lr scheduler
        # self.cfg.lr_decay = 0.8685113737513527

        self.lr_scheduler = ExponentialLR(
            self.optimizer, gamma=self.cfg.lr_decay)



    def parse_args(self, **kargs):
        # default parameters
        cfg = {
            # inference parameters
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            'adjust_scale': 0.001,
            # train parameters
            'initial_lr': 0.01,
            'lr_decay': 0.8685113737513527,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0,
            'prior_frames_num': 13 }

        for key, val in kargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('GenericDict', cfg.keys())(**cfg)

    # 当frame=0的时候调用次方法
    def init(self, image, box):
        image = np.asarray(image)  # 将PIL图片转换成numpy

        # convert box to 0-indexed and center based [y, x, h, w]
        # 之前的box是xywh，之后的是yxhw，后面的是先行后列的形式
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]  # center是先行号后列号

        # create hanning window
        # 'response_sz': 17,
        # 'response_up': 16,
        # np.hanning的作用是按照正弦生成一个序列，np.hanning(5)  [0. , 0.5, 1. , 0.5, 0. ]
        # >>> x = np.array(['a', 'b', 'c'], dtype=object)
        # >>> np.outer(x, [1, 2, 3])
        # array([[a, aa, aaa],
        #        [b, bb, bbb],
        #        [c, cc, ccc]], dtype=object)
        # 因此，这个窗口是起到平滑作用
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors   是三个尺度，从0.96， 1， 1.03
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        # 下面的z_sz和x_sz都是为没有乘以scale的情况下的大小，例如z_sz*scale=127
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz

        # exemplar image
        # 将第一帧的目标范围裁剪出来
        self.avg_color = np.mean(image, axis=(0, 1))
        exemplar_image = self._crop_and_resize(
            image, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            pad_color=self.avg_color)

        # exemplar features
        exemplar_image = torch.from_numpy(exemplar_image).to(
            self.device).permute([2, 0, 1]).unsqueeze(0).float()
        self.exemplar_image_seq = [exemplar_image] * self.para.prior_frames_num

        with torch.set_grad_enabled(False):
            self.net.eval()
            # self.kernel = self.net.feature(exemplar_image)
            self.kernel = self.cal_z_seq_feat(self.exemplar_image_seq)
        self.last_kernel = self.kernel

        # 计算第一帧的17*17的特征图的最大值,此时的response是没有进行归一化的,也就是没有减
        # 去最小值的那种归一化
        # 读取第一张图片
        first_frame_response_map = self.cal_first_frame_17_response_map(image)
        self.max_response_first_frame = first_frame_response_map.max()
        print(self.max_response_first_frame)


    # 计算第一帧的响应图,image是第一帧的图片
    def cal_first_frame_17_response_map(self, image):
        # search images
        # 这里是裁剪3张不同尺度的图片用于和exampler卷积
        instance_images = [self._crop_and_resize(
            image, self.center, self.x_sz,
            out_size=self.cfg.instance_sz,
            pad_color=self.avg_color)]
        instance_images = np.stack(instance_images, axis=0)
        # 这里要做一个辨析，什么时候用permute 什么时候用view
        instance_images = torch.from_numpy(instance_images).to(
            self.device).permute([0, 3, 1, 2]).float()

        # responses
        with torch.set_grad_enabled(False):
            self.net.eval()
            instances = self.net.feature(instance_images)  # 生成instances的embedding
            responses = F.conv2d(instances, self.kernel)*0.001   # 生成卷积
        responses = responses.squeeze(1).cpu().numpy()  # shape is [3 17 17]

        return responses.max()


    # z_seq是一个list,其中每一个里面都是一个可以卷积的
    def cal_z_seq_feat(self, z_seq):
        z_feat_cat = None
        for z in z_seq:
            z_feat = self.net.feature(z.to(self.device))
            if z_feat_cat is None:
                z_feat_cat = z_feat
            else:
                z_feat_cat = torch.cat((z_feat_cat, z_feat), 1)

        z_feat_deconv = self.net.deconv(z_feat_cat)
        return z_feat_deconv

    def update_kernel(self, image, box):
        image = np.asarray(image)  # 将PIL图片转换成numpy

        # convert box to 0-indexed and center based [y, x, h, w]
        # 之前的box是xywh，之后的是yxhw，后面的是先行后列的形式
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]  # center是先行号后列号

        # exemplar and search sizes
        # 下面的z_sz和x_sz都是为没有乘以scale的情况下的大小，例如z_sz*scale=127
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))

        # exemplar image
        # 将第一帧的目标范围裁剪出来
        self.avg_color = np.mean(image, axis=(0, 1))
        exemplar_image = self._crop_and_resize(
            image, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            pad_color=self.avg_color)

        # exemplar features
        exemplar_image = torch.from_numpy(exemplar_image).to(
            self.device).permute([2, 0, 1]).unsqueeze(0).float()
        # 下面这种更新方式只用最近的13帧来搞
        self.exemplar_image_seq.append(exemplar_image)
        self.exemplar_image_seq.pop(1)
        assert len(self.exemplar_image_seq) == self.para.prior_frames_num

        with torch.set_grad_enabled(False):
            self.net.eval()
            # self.kernel = self.net.feature(exemplar_image)
            self.kernel = self.cal_z_seq_feat(self.exemplar_image_seq)*(1-self.para.kernel_lr) + \
                          self.para.kernel_lr*self.last_kernel
            self.last_kernel = self.kernel


    # 当frame不等于第一帧时调用这个
    def update(self, image):
        img_to_show = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)  # 这个是为了用cv2显示用的,并不影响跟踪个结果
        image = np.asarray(image)  # image 是PIL类型的图片

        # search images
        # 这里是裁剪3张不同尺度的图片用于和exampler卷积
        instance_images = [self._crop_and_resize(
            image, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            pad_color=self.avg_color) for f in self.scale_factors]
        instance_images = np.stack(instance_images, axis=0)
        # 这里要做一个辨析，什么时候用permute 什么时候用view
        instance_images = torch.from_numpy(instance_images).to(
            self.device).permute([0, 3, 1, 2]).float()

        # responses
        with torch.set_grad_enabled(False):
            self.net.eval()
            instances = self.net.feature(instance_images)  # 生成instances的embedding
            responses = F.conv2d(instances, self.kernel) * 0.001  # 生成卷积
        responses = responses.squeeze(1).cpu().numpy()  # shape is [3 17 17]
        # 提取当前响应图的最大值
        max_response = responses.max()

        # upsample responses and penalize scale changes
        # self.upscale_sz = 272 将响应图resize到272大小，中间使用三先行插值
        responses = np.stack([cv2.resize(
            t, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC) for t in responses], axis=0)
        # self.cfg.scale_num = 3  # self.cfg.scale_penalty = 0.9745
        # 加入对尺度改变的惩罚
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty  # self.cfg.scale_penalty = 0.9745
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        # np.amax(a, axis=0) 表示对每一列求最大值，返回的值的尺寸是a的列数
        # np.argmax([8.793644, 8.928641, 8.579145]) = 1  # 返回最大值的索引
        # np.amax(responses, axis=(1, 2)) = [8.793644, 8.928641, 8.579145]  # 返回最大值
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]  # 找出具有最大值的那一层响应 是一个17 17的响应图
        response -= response.min()  # 所有的值
        response /= response.sum() + 1e-16  # 归一化
        # self.cfg.window_influence = 0.176
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        # response.argmax() 是从response中返回最大值的的顺序位置
        # np.unravel_index 是从顺序位置中根据shape的形状返回下标
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        # self.upscale_sz = 272
        # 这里的难道图像的中心点坐标是00？而不是x + w/2?
        # disp_in_response表示偏移中心的位移
        disp_in_response = np.array(loc) - self.upscale_sz // 2
        # cfg.total_stride = 8
        # cfg.response_up = 16
        # disp_in_instance 这里是从映射到了原图的坐标
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        # self.cfg.instance_sz = 255
        # self.x_sz = 3338.177
        # self.scale_factors[scale_id]表示缩放的尺度
        # disp_in_image = 到原图的坐标中相对位移
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image  # self.center 更新当前帧的目标中心

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        ################# 更新self.kernel ####################
        # 根据已跟踪到的box,求出这个box的
        if max_response > self.max_response_first_frame * self.para.update_template_threshold:
            self.update_kernel(image, box)
            # print('update-----',box)

        # showbb(img_to_show, box)
        return box

    def step(self, batch, backward=True, update_lr=False):
        # 每一次epoch运行一次self.lr_scheduler.step()
        # 这样学习率就会不断减少
        # 每一次epoch开始的时候updat_lr会为true，然后就会运行一步self.lr_scheduler.step()
        # 在当前epoch当中的学习率是不变的
        if backward:
            self.net.train()
            if update_lr:
                self.lr_scheduler.step()
        else:
            self.net.eval()

        z_seq = batch[0]
        x_target = batch[1]
        x_search = batch[2]
        # z = batch[0].to(self.device)  # 样本z放入gpu
        # x = batch[1].to(self.device)  # 样本x放入gpu
        # for i in range(z.shape[0]):
        #     showimg(z[i,:,:,:].to('cpu').data.numpy().transpose(1,2,0).astype('uint8'))
        #     showimg(x[i,:, :, :].to('cpu').data.numpy().transpose(1, 2, 0).astype('uint8'))
        with torch.set_grad_enabled(backward):
            z_deconv, x_target_feat, response17 = self.net(z_seq, x_target, x_search)
            # responses.size = 8 1 15 15
            # labels 的大小也是8 1 15 15
            # weights的大小也是8 1 15 15
            # 其中labels中间以2为半径内的label为1，其余的都为0，这里的是0和1
            # 而不是-1和1
            labels, weights = self._create_labels(response17.size())
            loss_cross_entropy = F.binary_cross_entropy_with_logits(
                response17, labels, weight=weights, size_average=True)
            loss_feat_regression = F.smooth_l1_loss(z_deconv, x_target_feat)

            total_loss = self.para.class_weight*loss_cross_entropy + \
                         self.para.regression_loss_weight * loss_feat_regression
            # g = make_dot(total_loss)
            # g.view()
            if backward:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        # 这里返回的是一个total loss
        return total_loss.item()

    # 在图像中，以center为中心裁剪size大小的图片，并且resize到out_size大小
    # 如果在center之外的地方，需要进行pad，pad的color是最后一个参数
    def _crop_and_resize(self, image, center, size, out_size, pad_color):
        # convert box to corners (0-indexed)
        size = round(size)
        corners = np.concatenate((
            np.round(center - (size - 1) / 2),
            np.round(center - (size - 1) / 2) + size))
        corners = np.round(corners).astype(int)

        # pad image if necessary
        pads = np.concatenate((
            -corners[:2], corners[2:] - image.shape[:2]))
        npad = max(0, int(pads.max()))
        if npad > 0:
            image = cv2.copyMakeBorder(
                image, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=pad_color)

        # crop image patch
        corners = (corners + npad).astype(int)
        patch = image[corners[0]:corners[2], corners[1]:corners[3]]

        # resize to out_size
        patch = cv2.resize(patch, (out_size, out_size))

        return patch

    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels, self.weights

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size  # n=8 c=1 h=15 w=15
        x = np.arange(w) - w // 2  # 除以2之后在floor
        y = np.arange(h) - h // 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        # self.cfg.r_pos = 16
        # self.cfg.total_stride = 8
        # self.cfg.r_neg = 0
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # pos/neg weights
        pos_num = np.sum(labels == 1)
        neg_num = np.sum(labels == 0)
        weights = np.zeros_like(labels)
        weights[labels == 1] = 0.5 / pos_num
        weights[labels == 0] = 0.5 / neg_num
        weights *= pos_num + neg_num

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        weights = weights.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))
        weights = np.tile(weights, [n, c, 1, 1])

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        self.weights = torch.from_numpy(weights).to(self.device).float()

        return self.labels, self.weights
