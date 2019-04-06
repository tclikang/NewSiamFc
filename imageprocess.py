import numpy as np
import cv2
import time

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





