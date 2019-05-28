# 使用一些图像处理的函数,以及读取文件的各种函数
import torch
import cv2
import os
import xml.etree.ElementTree as ET
import parameters
import numpy as np
import torch.nn.functional as f
import random


# 将文件夹中所有的文件根据文件名排序,并将文件名放入name_list中
def name_list_in_a_dir(dir_path):
    name_list = os.listdir(dir_path)
    name_list.sort()
    name_list = [os.path.join(dir_path, name) for name in name_list]
    return name_list

def read_anno_in_xml_file(anno_path):
    xml_tree = ET.parse(anno_path)  # 获得一棵树
    xml_root = xml_tree.getroot()  # 得到树根
    img_size = xml_root.find('size')
    img_width   = int(img_size.find('width').text)
    img_height  = int(img_size.find('height').text)
    img_channel = int(img_size.find('depth').text)
    img_size_para = [img_height, img_width, img_channel]  # (高度,宽度,深度)
    cls_name = []
    bb = []
    for object in xml_root.iter("object"):
        temp_cls_name = object.find("name").text
        bnd_root = object.find('bndbox')
        xmin = int(bnd_root.find('xmin').text)
        ymin = int(bnd_root.find('ymin').text)
        xmax = int(bnd_root.find('xmax').text)
        ymax = int(bnd_root.find('ymax').text)
        temp_bb = [xmin, ymin, xmax, ymax]
        cls_id = parameters.Parameters.clsname_to_label[temp_cls_name]
        cls_name.append(cls_id)
        bb.append(temp_bb)
    # 其中img_size_para的格式为(高度,宽度,深度)
    return img_size_para, cls_name, bb

# 讲最新的模型读入net,并删除老模型
def train_read_net(filepath, net, optimizer):
    # 读取训练数据
    if len(os.listdir(filepath)) > 0:  # 文件夹不为空
        model_list = os.listdir(filepath)
        model_list.sort()
        model_path = filepath + model_list[-1]
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model_para'])
        optimizer.load_state_dict(checkpoint['optimizer_para'])
        epoch = checkpoint['current_epoch']
        for name in model_list:
            file_name = os.path.join(filepath, name)
            os.remove(file_name)
        save_net(filepath, net, optimizer, 0, 0)
        return epoch
    else:
        return 0



# 保存网络和优化的参数放入文件中
def save_net(filepath, net, optimizer, epoch, step):
    torch.save({'model_para': net.state_dict(),
                'current_epoch': epoch,
                'optimizer_para': optimizer.state_dict()}, '{}E{:0>2d}S{:0>10}.pkl'.format(filepath, epoch, step))

# 讲filepath中的模型读入网络,但是并不删除老模型
def test_read_net(filepath, net):
    if len(os.listdir(filepath)) > 0:  # 文件夹不为空
        model_list = os.listdir(filepath)
        model_list.sort()
        model_path = filepath + model_list[-1]
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model_para'])

