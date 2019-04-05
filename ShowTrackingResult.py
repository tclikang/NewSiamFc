import cv2
import os
import glob
import sys
import imageprocess

img_path = '/home/fanfu/data/GOT-10k/test/GOT-10k_Test_000017'
result_path = '/home/fanfu/PycharmProjects/SimpleSiamFC/results/GOT-10k/SiamFC/GOT-10k_Test_000017/GOT-10k_Test_000017_001.txt'



if __name__ == '__main__':
    img_list = glob.glob(os.path.join(img_path, "*.jpg"))
    img_list.sort()

    result = []
    with open(result_path, 'r') as f:
        for line in f:
            result.append(list(line.strip('\n').split(',')))
    print(result)
    for (i, res) in zip(img_list, result):
        img = cv2.imread(i)
        x,y,w,h = [int(float(x)) for x in res]
        imageprocess.showbb(img, [x,y,w,h])



    print(img_list)





















