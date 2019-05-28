import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import utils
import visdom
from test_bb_reg.dataset import *
import test_bb_reg.bb_tools as tb
import test_bb_reg.utils as tu
import torch.optim as optim
import test_bb_reg.visualize_tools as tv
import cv2
import test_bb_reg.file_tools as file_tools
from torch.optim.lr_scheduler import ExponentialLR



class MyNet(nn.Module):

    def __init__(self, **kargs):
        super(MyNet, self).__init__()
        # 这里讲一下conv2中的group参数
        # 假设当前的输入channel数为8， 输出为12，卷积核大小为3，
        # 则当group=1时， 总共有12个神经元，每个神经元的参数大小为3*3*8，每个神经元生成一个channel
        # 假设当group=4的时候，每一组的神经元输入channel=2，输出=3，总共仍然有12个神经元，只不过每个神经元的大小为3*3*2，而不是3*3*8
        # Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), groups=2)
        # self.feature[4].weight.data.shape = torch.Size([256, 48, 5, 5])
        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
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
        self.relu = nn.ReLU(inplace=True)
        # self.ave_pool = nn.AvgPool2d(6)
        self.linear1 = nn.Linear(512*6*6, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, 4)
        # 初始化权值
        self._initialize_weights()

    def forward(self, z_img, x_img):
        # z_img and x_img are 127*127*6
        # bb type is [xmin ymin xmax ymax]
        z = self.feature(z_img)
        x = self.feature(x_img)
        concat_zx = torch.cat((z,x), 1)  # 8*512*6*6
        # out = self.ave_pool(concat_zx)
        out = self.relu(self.linear1(concat_zx.view(concat_zx.shape[0], -1)))
        out = self.relu(self.linear2(out))
        out = self.linear3(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, mode='fan_out',
                                      nonlinearity='relu')
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight.data, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == '__main__':
    # 初始化可视化工具
    vis = visdom.Visdom()
    model_file = '/home/fanfu/PycharmProjects/SimpleSiamFC/test_bb_reg/model_file/'
    # 初始化模型
    model = MyNet()
    model = model.to(model.device)
    # 初始化epoch
    current_epoch = 0
    # 初始化优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,  # self.cfg.initial_lr = 0.01
        weight_decay=0.0005,  # self.cfg.weight_decay = 0.0005
        momentum=0.9)  # self.cfg.momentum = 0.9
    # lr_scheduler.step()
    # 预读取模型
    current_epoch = file_tools.train_read_net(model_file, model, optimizer)
    # 初始化优化器的参数更新
    lr_scheduler = ExponentialLR(
        optimizer, gamma=0.8685113737513527, last_epoch = -1)
    for i in range(current_epoch):
        lr_scheduler.step()
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
        pair_dataset, batch_size=8, shuffle=True, pin_memory=cuda,
        drop_last=True, num_workers=8)

    show_loss = 0
    for epoch in range(current_epoch, 25):
        lr_scheduler.step()
        for step, (z_img_files, x_img_files, z_img, x_img,
                   z_loc, x_loc, x_loc_rao, reg_score) in enumerate(loader):
            # 将需要计算的图片放入cuda中
            z_img = z_img.cuda()
            x_img = x_img.cuda()
            z_loc = z_loc.float().cuda()
            x_loc = x_loc.float().cuda()
            x_loc_rao = x_loc_rao.float().cuda()
            # 将图片输入网络,得到reg的分数
            reg_net = model(z_img, x_img).cuda()  # 得到的是[8 4]的regression
            new_bb = tu.do_reg_net(x_loc_rao, reg_net)  # new_bb也是[8 4]的包围框
            iou_loss = tu.compute_iou_loss_from_two_tensor(new_bb, x_loc)  # 都是xmin ymin的参数
            # 计算loss
            show_loss += iou_loss.item()
            # 优化参数
            optimizer.zero_grad()
            iou_loss.backward()
            optimizer.step()
            # ------------------相关作图代码-------------------------------------
            if step % 20 ==0:
                vis.close()  # 关闭显示
                z_img_show = F.to_tensor(tv.read_img(z_img_files[0])).unsqueeze(0).numpy()*255.0
                x_img_show = F.to_tensor(tv.read_img(x_img_files[0])).unsqueeze(0).numpy()*255.0

                z_img_show = tv.show_torch_img_with_bblist(z_img_show, [z_loc[0]])
                x_img_show1 = tv.show_torch_img_with_bblist(x_img_show, [x_loc_rao[0]])
                x_img_show2 = tv.show_torch_img_with_bblist(x_img_show, [new_bb[0]])
                vis.images(np.vstack((z_img_show, x_img_show1, x_img_show2)),
                                       opts=dict(title='origin image', caption='How random.'))
                print('step is {}, ave 20 loss is {}'.format(step, show_loss/20))
                show_loss = 0
        file_tools.save_net(model_file, model, optimizer, epoch, step=0)















