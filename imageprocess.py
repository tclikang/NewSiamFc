import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from matplotlib import axes
import cv2

# 画图像包围框
def showbb(img, boundingbox):
    #img 是一种numpy格式的图像
    x,y,w,h = boundingbox
    cv2.rectangle(img, (int(x),int(y)),(int(x+w), int(y+h)), (0,0,255))
    showimg(img)

# 画图像
def showimg(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)

# 将responsemap resize到w,h大小,然后画出热度图
def createheatmap(responsemap):
    show_map = (responsemap-responsemap.min())/responsemap.max()
    heatmap_img = cv2.applyColorMap(np.uint8(show_map*255), cv2.COLORMAP_JET)
    return heatmap_img


def showheatmap(origin_img, heat_map, alpha=0.5):
    # origin_img 是原始图片
    # heat_map 是需要覆盖上的热度图
    # origin_map 和 heat_map的大小应该一致
    dst = cv2.addWeighted(origin_img, alpha, heat_map, (1-alpha), 0)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)

if __name__ == '__main__':
    origin_img = cv2.imread('/home/fanfu/data/OTB/Basketball/img/0001.jpg')
    heat_map = cv2.imread('/home/fanfu/data/OTB/Coke/img/0011.jpg')
    heat_map = cv2.resize(heat_map, (origin_img.shape[1], origin_img.shape[0]),interpolation=cv2.INTER_CUBIC)
    showheatmap(origin_img, heat_map)







