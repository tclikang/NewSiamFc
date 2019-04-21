import numpy as np
import imageprocess
import cv2

# img是从cv2中读取的图像
# heatmap是热图图像
# center是热图图像中心在img中的中心
# img heatmap center 的坐标都是以左上角为0开始的
def drawheatmap(img, heatmap, center):
    borderType = cv2.BORDER_CONSTANT # padding使用固定像素
    pad_value = [0,0,0]
    img_h, img_w,_ = img.shape
    hm_h, hm_w, = heatmap.shape
    c_y, c_x = center
    half_hm_h = hm_h // 2
    half_hm_w = hm_w // 2
    # 从上面截断heatmap
    if half_hm_h > c_y:
        jieduan= (half_hm_h - c_y).astype(int)
        heatmap = heatmap[jieduan:, :]  # 对图像截断
    else: # 在上面padding
        top_pad = (c_y - half_hm_h).astype(int)
        heatmap = cv2.copyMakeBorder(heatmap, top=top_pad, bottom=0, left=0, right=0,
                                     borderType=borderType, dst=None, value=pad_value)

    # 在下面截断
    if half_hm_h + c_y > img_h:
        bottom_crop = ((half_hm_h + c_y) - img_h).astype(int)
        heatmap = heatmap[:- bottom_crop, :]
    else: # 在下面padding
        bottom_padding = (img_h - (half_hm_h + c_y)).astype(int)
        heatmap = cv2.copyMakeBorder(heatmap, top=0, bottom=bottom_padding, left=0, right=0,
                                     borderType=borderType, dst=None, value=pad_value)

    # 在左边截断
    if half_hm_w > c_x:
        left_crop = (half_hm_w - c_x).astype(int)
        heatmap = heatmap[left_crop:,:]
    else:
        left_pad = (c_x - half_hm_w).astype(int)
        heatmap = cv2.copyMakeBorder(heatmap, top=0, bottom=0, left=left_pad, right=0,
                                     borderType=borderType, dst=None, value=pad_value)

    # 在右面截断
    if half_hm_w + c_x > img_w:
        right_crop = ((half_hm_w + c_x) - img_w).astype(int)
        heatmap = heatmap[:, :-right_crop]
    else:  # 在右面padding
        right_padding = (img_w - (half_hm_w + c_x)).astype(int)
        heatmap = cv2.copyMakeBorder(heatmap, top=0, bottom=0, left=0, right=right_padding,
                                     borderType=borderType, dst=None, value=pad_value)
    # 到这一步已经讲heatmap padding到与原图一样大小
    heatmap = cv2.resize(heatmap, (img_w, img_h))
    heatmap = imageprocess.createheatmap(heatmap)
    blendimg = imageprocess.showheatmap(img, heatmap)




if __name__ == '__main__':
    pass






