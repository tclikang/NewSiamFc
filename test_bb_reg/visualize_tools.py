import torch
import numpy as np
import cv2

# imgs 是 nchw类型的矩阵
def imgs_to_imglist(imgs):
    img_list = []
    for img in imgs:
        img_list.append(img)
    return img_list

# 将chw类型的矩阵转化为hwc类型的
def chwimg_to_hwcimg(chwimg):
    hwcimg_list = []
    for img in chwimg:
        hwcimg_list.append(img.swapaxes(0,1).swapaxes(1,2))
    return hwcimg_list


# 将chw类型的矩阵转化为hwc类型的
def hwcimg_to_chwimg(chwimg):
    hwcimg_list = []
    for img in chwimg:
        hwcimg_list.append(img.swapaxes(1,2).swapaxes(0,1))
    return hwcimg_list

# 将bgr 转换成rgb
def bgr_to_rgb(img_list):
    rgb_list = []
    for img in img_list:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_list.append(img_rgb)
    return rgb_list

# 将rgb 转换成 bgr
def rgb_to_bgr(img_list):
    rgb_list = []
    for img in img_list:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        rgb_list.append(img_rgb)
    return rgb_list

#chwlist 转换成nchw
def list_to_torch(chw_list):
    return np.array(chw_list)

# torch_img是ncwh类型的第一维是1
# bblist 是xmin格式的包围框
def show_torch_img_with_bblist(torch_img, bblist):
    out_list = imgs_to_imglist(torch_img)
    out_list = chwimg_to_hwcimg(out_list)
    out_list = [showimg_with_bb(out_list[0], bblist)]
    # showimg(out_list[0])
    out_list = bgr_to_rgb(out_list)
    out_list = hwcimg_to_chwimg(out_list)
    nchw_img = list_to_torch(out_list)
    return nchw_img

# img is read by read_img
# bb is a list of bounding box
def showimg_with_bb(img, bb_list, type='corner',color = (0,0,255)):
    if type is 'xywh':
        for bb in bb_list:
            x, y, w, h = bb
            img = cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color)
    if type is 'corner':
        for bb in bb_list:
            xmin, ymin, xmax, ymax = bb
            img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color)
    return img.get()

def showimg(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)

# 所有的图像格式为[高度,宽度,通道数]
# 图像左上角的起点坐标为0 0,右下角结束坐标为width-1, height-1
def read_img(img_path):
    img = cv2.imread(img_path)
    return img


# 将img缩放到dis_h,dis_w的尺寸,并返回缩放比
# img is [h w 3]的numpy类型
def img_resize(img, dis_h, dis_w):
    img_h, img_w, _ = img.shape
    img_after_resize = cv2.resize(img, (dis_w, dis_h))
    scale_h = dis_h / img_h
    scale_w = dis_w / img_w
    return img_after_resize, scale_h, scale_w


# bb是一个list,格式为xmin, ymin, xmax, ymax
# 将其缩放到scale_h,scale_w的倍数
# img_h img_w是缩放后的图像尺寸
def anno_resize(bb, scale_h, scale_w, img_h, img_w):
    bb_resize_list = []
    for b in bb:
        temp_xmin = max(int(b[0] * scale_w), 0)
        temp_xmax = min(int(b[2] * scale_w), img_w-1)
        temp_ymin = max(int(b[1] * scale_h), 0)
        temp_ymax = min(int(b[3] * scale_h), img_h-1)
        bb_resize_list.append([temp_xmin, temp_ymin, temp_xmax, temp_ymax])
    return bb_resize_list


# img is read by read_img
# bb is a list of bounding box
def showimg_with_bb(img, bb_list, type='corner',color = (255,0,0)):
    if type is 'xywh':
        for bb in bb_list:
            x, y, w, h = bb
            img = cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color)
    if type is 'corner':
        for bb in bb_list:
            xmin, ymin, xmax, ymax = bb
            img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 8)
            img = cv2.resize(img, (500,500),interpolation=cv2.INTER_CUBIC)
    # showimg(img)
    return img.get()


def showimg(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)