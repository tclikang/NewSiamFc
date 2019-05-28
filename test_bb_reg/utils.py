import numpy as np
import torch
import test_bb_reg.bb_tools as bbt
from torchvision.transforms import Compose, CenterCrop, RandomCrop, ToTensor


# *****代码正确
# 进行一次bb reg
# model是网络模型
# z_img是第一帧的图像,用Image.open读取 z_img = Image.open(img_files[z_index])
# z_loc是xmin ymin格式的位置
# x_img是需要做回归的图像
def do_bbreg_according_to_network(model, z_img, z_loc, x_img, x_loc):
    # 对图像进行转换
    # augmentation for exemplar and instance images
    transform_z = Compose([
        CenterCrop(127),  # cfg.exemplar_sz = 127  然后在中心裁剪成127 * 127
        ToTensor()])
    transform_x = Compose([
        CenterCrop(127),  # cfg.exemplar_sz = 127  然后在中心裁剪成127 * 127
        ToTensor()])
    # 转换bb的格式,从xmin ymin转换成xywh topleft
    # -----------裁剪x,z,将它们放入tensor
    z_img_crop = bbt.crop_and_resize_from_minmax(z_img, z_loc)
    z_img_crop = transform_z(z_img_crop) * 255.0
    x_img_crop = bbt.crop_and_resize_from_minmax(x_img, x_loc)
    x_img_crop = transform_x(x_img_crop) * 255.0
    # 转化成能处理的格式 [n c h w]
    z_img_crop = z_img_crop.unsqueeze(0).cuda()  # 转化成网络的输入格式
    x_img_crop = x_img_crop.unsqueeze(0).cuda()  # 转化成网络输入的格式
    # 进行 bbreg
    reg_score = model(z_img_crop, x_img_crop)  # 得到的是[8 4]的regression
    # -----------------强行令w和h的比例相等,实验完成就删除-------------------------------
    t1 = abs(reg_score[0, 2])
    t2 = abs(reg_score[0, 3])
    if t1 < t2:
        temp = reg_score[0, 2]
    else:
        temp = reg_score[0, 3]
    reg_score[0, 2] = reg_score[0, 3] = temp
    # -----------------强行令w和h的比例相等-------------------------------
    reg_score = reg_score.cpu().numpy()[0]
    x_new_loc = bbt.bbreg_from_minmax_to_minmax(x_loc, reg_score)  # new_bb也是[8 4]的包围框
    return x_new_loc  # minmax类型的


# *****代码正确
# anno的格式是topleft类型
# anno1是gt
# anno2是歪的
def get_tx_ty_tw_th_from_two_anno(anno1, anno2):
    anno1_minmax = bbt.bb_from_xywh_topleft_to_minmax(anno1)
    anno2_minmax = bbt.bb_from_xywh_topleft_to_minmax(anno2)
    reg = bbt.get_tx_ty_tw_th_from_two_bb(anno1_minmax, anno2_minmax)
    return reg


# *****代码正确
# bb1 bb2是两个[8,4]的tensor
# 计算它们之间的iouloss
def compute_iou_loss_from_two_tensor(bb1, bb2):
    loss = torch.zeros(1).cuda()
    for b1, b2 in zip(bb1, bb2):
        loss = loss + (-torch.log(compute_iou(b1, b2) + 1e-16))
    loss = loss / bb1.shape[0]
    return loss


# *****代码正确
# 计算iou的函数
def compute_iou(box1, box2, wh=False):
    """
    compute the iou of two boxes.
    Args:
        box1, box2: [xmin, ymin, xmax, ymax] (wh=False) or [xcenter, ycenter, w, h] (wh=True)
        wh: the format of coordinate.
    Return:
        iou: iou of box1 and box2.
    """
    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0] - box1[2] / 2.0), int(box1[1] - box1[3] / 2.0)
        xmax1, ymax1 = int(box1[0] + box1[2] / 2.0), int(box1[1] + box1[3] / 2.0)
        xmin2, ymin2 = int(box2[0] - box2[2] / 2.0), int(box2[1] - box2[3] / 2.0)
        xmax2, ymax2 = int(box2[0] + box2[2] / 2.0), int(box2[1] + box2[3] / 2.0)

    ## 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = torch.max(xmin1, xmin2)
    yy1 = torch.max(ymin1, ymin2)
    xx2 = torch.min(xmax1, xmax2)
    yy2 = torch.min(ymax1, ymax2)

    ## 计算两个矩形框面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    inter_area = (torch.max(torch.tensor(0.).cuda(), xx2 - xx1)) * (
        torch.max(torch.tensor(0.).cuda(), yy2 - yy1))  # 计算交集面积
    iou = inter_area / (area1 + area2 - inter_area + 1e-16)  # 计算交并比

    return iou


# *****代码正确
# bbs 是[8 4]的包围框 xmin格式的
# regs[8 4]的包围框 tx格式的
def do_reg_net(bbs, regs):
    bbs_new = []
    for i in range(bbs.shape[0]):
        bbs_new.append(bbreg_from_minmax_to_minmax(bbs[i], regs[i]))
    return torch.stack(bbs_new)


# *****代码正确
# 对xmin格式的包围框做回归 都是numpy格式
# anchor 是一个xmin,ymin,xmax,ymax的一维向量
# reg_score是一个tx,ty,tw,th的变换
# 返回一个变换后的xmin格式的包围框
def bbreg_from_minmax_to_minmax(anchor, reg_score):
    anchor_xywh = bb_from_minmax_to_xywhcenter(anchor)
    anchor_xywh_after_reg = bbreg_xywhbb_trans(anchor_xywh, reg_score)
    anchor_minmax = bb_from_xywhcenter_to_minmax(anchor_xywh_after_reg)
    return anchor_minmax


# *****代码正确
# 从xmin ymin xmax ymax 转换成xcenter ycenter w h
def bb_from_minmax_to_xywhcenter(bb):
    xmin = bb[0]
    ymin = bb[1]
    xmax = bb[2]
    ymax = bb[3]
    w = xmax - xmin + 1
    h = ymax - ymin + 1
    x_center = xmin + (w - 1) / 2
    y_center = ymin + (h - 1) / 2
    return torch.stack((x_center, y_center, w, h))


# *****代码正确
# bb是一个xywh格式的包围框
# reg_score是一个tx,ty,tw,th的变换
# 返回一个xywh格式的包围框
def bbreg_xywhbb_trans(bb, reg_score):
    x, y, w, h = bb
    dx, dy, dw, dh = reg_score
    res_x = w * dx + x
    res_y = h * dy + y
    res_w = w * torch.exp(dw)
    res_h = h * torch.exp(dh)
    return torch.stack((res_x, res_y, res_w, res_h))


# *****代码正确
# 从xcenter ycenter w h转换成xmin ymin xmax ymax
def bb_from_xywhcenter_to_minmax(bb):
    x_center = bb[0]
    y_center = bb[1]
    w = bb[2]
    h = bb[3]
    xmin = x_center - (w - 1) / 2
    ymin = y_center - (h - 1) / 2
    xmax = xmin + w - 1
    ymax = ymin + h - 1
    return torch.stack((xmin, ymin, xmax, ymax))


if __name__ == '__main__':
    bb1 = torch.tensor([100, 50, 500, 300]).float()
    reg = torch.tensor([0.13105473343531948, -0.033663478107132244, -0.07829068051325883, 0.030208200061014656]).float()

    print(bbreg_from_minmax_to_minmax(bb1, reg))
    pass
