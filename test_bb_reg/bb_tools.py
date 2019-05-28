# 这是用于有关bb相关的工具函数
import numpy as np
import cv2
import random
import torch
from PIL import Image, ImageStat, ImageOps
from torchvision.transforms import Compose, CenterCrop, RandomCrop, ToTensor

# ------------------------图像格式统一简介---------------------------------------
'''xmin ymin xmax ymax的起始下标为0,
xywh lefttop 中的xy是左上角的位置,并且起始坐标点为1,
xywh center中的xy是中心位置,起始坐标点为0
'''

# *****代码正确
# anno是一个xmin ymin xmax ymax的包围框
# w,h是一个图像的宽和高
# 扰动之后不能够超出图像范围
# 返回一个xmin ymin xman ymax的扰动后的包围框
# alpha表示xmin  ymin 的变动范围在-alpha +alpha之内均匀采样
# beta表示宽度的缩放比例
def raodong_bb(anno, img_w, img_h, alpha=0.2, beta=0.3):
    xc, yc, w, h = bb_from_minmax_to_xywhcenter(anno)
    tx = random.uniform(-alpha, alpha)
    ty = random.uniform(-alpha, alpha)
    tw = random.uniform(np.log(1 - beta), np.log(1 + beta))
    th = random.uniform(np.log(1 - beta), np.log(1 + beta))
    # print(tx, ty, tw, th)
    xc_new = xc + tx * w
    yc_new = yc + ty * h
    w_new = w * np.exp(tw)
    h_new = h * np.exp(th)

    xmin, ymin, xmax, ymax = bb_from_xywhcenter_to_minmax([xc_new, yc_new, w_new, h_new])
    xmin = np.clip(xmin, 0, img_w - 1)
    ymin = np.clip(ymin, 0, img_h - 1)
    xmax = np.clip(xmax, 0, img_w - 1)
    ymax = np.clip(ymax, 0, img_h - 1)

    return [xmin, ymin, xmax, ymax]


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
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    ## 计算两个矩形框面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))  # 计算交集面积
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)  # 计算交并比

    return iou

# *****代码正确,通过验证
# anchor_array, gt_array是一个二维数组,每一行是一个xmin,ymin,xmax,ymax
# reg_array是一个二维数组,每一行都是一个reg
def get_tx_ty_tw_th_list_from_two_bblist(gt_array, anchor_array):
    reg_array = []
    for gt, anchor in zip(gt_array, anchor_array):
        reg_array.append(get_tx_ty_tw_th_from_two_bb(gt, anchor))

    return np.vstack(reg_array)


# *****代码正确
# gt, anchor 是xmin,ymin,xmax,ymax类型的包围框
# 都是numpy类型的数据,第一个参数是想要变换到的位置
def get_tx_ty_tw_th_from_two_bb(gt, anchor):
    gt_xywh = bb_from_minmax_to_xywhcenter(gt)
    anchor_xywh = bb_from_minmax_to_xywhcenter(anchor)
    tx = (gt_xywh[0] - anchor_xywh[0]) / anchor_xywh[2]
    ty = (gt_xywh[1] - anchor_xywh[1]) / anchor_xywh[3]
    tw = np.log(gt_xywh[2] / anchor_xywh[2])
    th = np.log(gt_xywh[3] / anchor_xywh[3])
    return np.array([tx, ty, tw, th])


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
    return np.array([x_center, y_center, w, h])


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
    return np.array([xmin, ymin, xmax, ymax])

# *****代码正确
# 从xmin ymin xmax ymax 转换成xmin ymin w h
# 传统的xywh下标是从1开始的
def bb_from_minmax_to_xywh_topleft(bb):
    xmin = bb[0]
    ymin = bb[1]
    xmax = bb[2]
    ymax = bb[3]
    w = xmax - xmin + 1
    h = ymax - ymin + 1
    x = xmin + 1
    y = ymin + 1
    return np.array([x, y, w, h])

# *****代码正确
# 从xmin ymin w h转换成xmin ymin xmax ymax
# 传统的xywh下标是从1开始的
def bb_from_xywh_topleft_to_minmax(bb):
    x = bb[0]
    y = bb[1]
    w = bb[2]
    h = bb[3]
    xmin = x - 1
    ymin = y - 1
    xmax = xmin + w - 1
    ymax = ymin + h - 1
    return np.array([xmin, ymin, xmax, ymax])

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
# bb是一个xywh center格式的包围框
# reg_score是一个tx,ty,tw,th的变换
# 返回一个xywh格式的包围框
def bbreg_xywhbb_trans(bb, reg_score):
    x, y, w, h = bb  # center 类型
    dx, dy, dw, dh = reg_score
    res_x = w * dx + x
    res_y = h * dy + y
    res_w = w * np.exp(dw)
    res_h = h * np.exp(dh)
    return np.array([res_x, res_y, res_w, res_h])

# *****代码正确
# box 是minmax类型的
# image是 PIL图像
def crop_and_resize_from_minmax(image, box):
    # 将minmax转成topleft
    box_topleft = bb_from_minmax_to_xywh_topleft(box)
    return crop_and_resize(image, box_topleft)


# *****代码正确
# image是PIL图像，box是目标的xywh
# 这里的box中的坐标x是从1开始记录的，下面要从0的索引开始
def crop_and_resize(image, box):
    # convert box to 0-indexed and center based
    box = np.array([
        box[0] - 1 + (box[2] - 1) / 2,
        box[1] - 1 + (box[3] - 1) / 2,
        box[2], box[3]], dtype=np.float32)
    center, target_sz = box[:2], box[2:]  # center是目标中心索引，从0开始 target_sz是尺度

    # exemplar and search sizes
    # self.cfg.context = 0.5
    # np.sum(target_sz)是将目标的长宽相加
    context = 0.5 * np.sum(target_sz)  # (w+h)*0.5
    # 这里z_sz乘以scale之后才是127的大小
    # 这里x_sz乘以scale之后才是255的大小
    z_sz = np.sqrt(np.prod(target_sz + context))  # eq.7     s*z_sz = 127
    x_sz = z_sz * 255 / 127.0  # s*x_sz = 255

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
    # out_size = (self.cfg.instance_sz, self.cfg.instance_sz)
    out_size = (255, 255)
    patch = patch.resize(out_size, Image.BILINEAR)
    # 最后返回的的是一个255大小的图片，裁剪后的255大小的图片
    return patch


if __name__ == '__main__':
    bb1 = np.array([100, 200, 400, 600])
    bb2 = np.array([150, 250, 600, 500])
    print(bb_from_xywhcenter_to_minmax(bb1))



    # reg = [0.13105473343531948, -0.033663478107132244, -0.07829068051325883, 0.030208200061014656]
    # anchor = [100, 50, 500, 300]
    # print(bbreg_from_minmax_to_minmax(anchor, reg))
    # # tt = [0.13105473343531948, -0.033663478107132244, -0.07829068051325883, 0.030208200061014656]
    # # raodonghou = [[167.65148101271993, 37.70149532341679, 537.4544152024063, 295.39943866680284]]
    # #temp = raodong_bb([100, 50, 500, 300], 2000, 2000)
    # #print(temp)
    # gt_array = [[167.65148101271993, 37.70149532341679, 537.4544152024063, 295.39943866680284],[167.65148101271993, 37.70149532341679, 537.4544152024063, 295.39943866680284]]
    # anchor_array = [[100, 50, 500, 300],[100, 50, 500, 300]]
    # print(get_tx_ty_tw_th_list_from_two_bblist(gt_array, anchor_array))
    pass
