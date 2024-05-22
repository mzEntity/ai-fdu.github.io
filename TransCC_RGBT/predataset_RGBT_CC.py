import glob
import math
import os
import torch
import cv2
import h5py
import numpy as np
import scipy.io as io
import scipy.spatial
from scipy.ndimage.filters import gaussian_filter
import random

# 先224*224
'''set your data path'''
root = './rgbtcc_fdu/'

options = ["train", "test", "val"]

for option in options:
    dir = f"{option}/"
    rgbt_cc = os.path.join(root, dir)

    print(f"start process {option}")
    if not os.path.exists(rgbt_cc.replace(option, f'new_{option}_224')):
        os.makedirs(rgbt_cc.replace(option, f'new_{option}_224'))

    img_paths = []
    for img_path in glob.glob(os.path.join(rgbt_cc, '*RGB.jpg')):
        img_paths.append(img_path)

    img_paths.sort()

    np.random.seed(0)
    random.seed(0)
    for img_path in img_paths:
        Img_data = cv2.imread(img_path)
        T_data = cv2.imread(img_path.replace('_RGB', '_T'))

        gt_exist = os.path.exists(img_path.replace('_RGB.jpg', '_GT.npy'))
        if gt_exist:
            Gt_data = np.load(img_path.replace('_RGB.jpg', '_GT.npy'))
            # 448和672
            rate = 1
            rate_1 = 1
            rate_2 = 1
            flag = 0
            if Img_data.shape[1] >= Img_data.shape[0]:  # 后面的大
                rate_1 = 672.0 / Img_data.shape[1]
                rate_2 = 448.0 / Img_data.shape[0]
                Img_data = cv2.resize(Img_data, (0, 0), fx=rate_1, fy=rate_2)
                T_data = cv2.resize(T_data, (0, 0), fx=rate_1, fy=rate_2)
                Gt_data[:, 0] = Gt_data[:, 0] * rate_1
                Gt_data[:, 1] = Gt_data[:, 1] * rate_2                
            elif Img_data.shape[0] > Img_data.shape[1]:  # 前面的大
                rate_1 = 672.0 / Img_data.shape[0]
                rate_2 = 448.0 / Img_data.shape[1]
                Img_data = cv2.resize(Img_data, (0, 0), fx=rate_2, fy=rate_1)
                T_data = cv2.resize(T_data, (0, 0), fx=rate_2, fy=rate_1)
                Gt_data[:, 0] = Gt_data[:, 0] * rate_2 # 对应的坐标进行扩大映射
                Gt_data[:, 1] = Gt_data[:, 1] * rate_1 # 对应的坐标进行扩大映射
            # else: # 训练才需要位置，验证，测试不需要
            img_path = img_path.replace(option, f"new_{option}_224")
            print(img_path)
            T_path = img_path.replace('_RGB','_T')
            gt_save_path = img_path.replace('_RGB.jpg', '_GT.npy')
            cv2.imwrite(img_path, Img_data)
            cv2.imwrite(T_path, T_data)
            np.save(gt_save_path, Gt_data)
        else:
            # 448和672
            rate = 1
            rate_1 = 1
            rate_2 = 1
            flag = 0
            if Img_data.shape[1] >= Img_data.shape[0]:  # 后面的大
                rate_1 = 672.0 / Img_data.shape[1]
                rate_2 = 448.0 / Img_data.shape[0]
                Img_data = cv2.resize(Img_data, (0, 0), fx=rate_1, fy=rate_2)
                T_data = cv2.resize(T_data, (0, 0), fx=rate_1, fy=rate_2)
            elif Img_data.shape[0] > Img_data.shape[1]:  # 前面的大
                rate_1 = 672.0 / Img_data.shape[0]
                rate_2 = 448.0 / Img_data.shape[1]
                Img_data = cv2.resize(Img_data, (0, 0), fx=rate_2, fy=rate_1)
                T_data = cv2.resize(T_data, (0, 0), fx=rate_2, fy=rate_1)
            # else: # 训练才需要位置，验证，测试不需要
            img_path = img_path.replace(option, f"new_{option}_224")
            print(img_path)
            T_path = img_path.replace('_RGB','_T')
            cv2.imwrite(img_path, Img_data)
            cv2.imwrite(T_path, T_data)
            
