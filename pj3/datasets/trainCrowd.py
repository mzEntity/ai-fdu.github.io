import os
import torch
import torch.nn as nn
from torchvision import transforms
import time
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import h5py
import cv2
import shutil


# 从im_h * im_w中切割出crop_h * crop_w大小的图片，返回图片左上角（四个点中较小的的x和y）坐标
def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w

def gen_discrete_map(im_height, im_width, points):
    """
        func: generate the discrete map.
        points: [num_gt, 2], for each row: [width, height]
        """
    discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
    h, w = discrete_map.shape[:2]
    num_gt = points.shape[0]
    if num_gt == 0:
        return discrete_map
    # fast create discrete map
    # points_np = np.array(points).round().astype(int)
    # p_h = np.minimum(points_np[:, 1], np.array([h-1]*num_gt).astype(int))
    # p_w = np.minimum(points_np[:, 0], np.array([w-1]*num_gt).astype(int))
    # p_index = torch.from_numpy(p_h* im_width + p_w)
    # discrete_map = torch.zeros(im_width * im_height).scatter_add_(0, index=p_index, src=torch.ones(im_width*im_height)).view(im_height, im_width).long().numpy()

    ''' 
    slow method
    '''
    for p in points:
        p = np.round(p).astype(int)
        p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
        discrete_map[p[0], p[1]] += 1
    assert np.sum(discrete_map) == num_gt
    return discrete_map

def load_RGB_or_Thermal(img_path):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if width > height:
        rate_0 =672.0 / width
        rate_1 =448.0 / height
        img =cv2.resize(img, (0, 0), fx=rate_0, fy=rate_1)
    else:
        rate_0 = 448.0 / width
        rate_1 = 672.0 / height
        img =cv2.resize(img, (0, 0), fx=rate_0, fy=rate_1)
    return img

def load_Target(gt_path):
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    # target = cv2.resize(target, (target.shape[1]//8, target.shape[0]//8), interpolation=cv2.INTER_CUBIC)*64
    return target

class TrainCrowd(Dataset):
    def __init__(self, shape=None, batch_size=1, num_workers=4):
        self.RGB_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.407, 0.389, 0.396],
                std=[0.241, 0.246, 0.242]
            ),
        ])
        self.T_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.492, 0.168, 0.430],
                std=[0.317, 0.174, 0.191]
            )
        ])
        
        self.c_size = 224
        self.d_ratio = 8
        
        self.shape = shape
        self.batch_size = batch_size
        self.num_workers = num_workers
         
        self.img_dir = "./dataset/train/rgb/"
        self.gt_dir = "./dataset/train/hdf5s/"
        
        self.rgb_img_paths = []
        self.thermal_img_paths = []
        for filename in os.listdir(self.img_dir):
            if not filename.endswith('.jpg'):
                continue
            rgb_file_path = os.path.join(self.img_dir, filename)
            thermal_file_path = rgb_file_path.replace("rgb/", "tir/").replace(".jpg", "R.jpg")
            
            self.rgb_img_paths.append(rgb_file_path)
            self.thermal_img_paths.append(thermal_file_path)
            
        self.img_paths = list(zip(self.rgb_img_paths, self.thermal_img_paths))
        self.nSamples = len(self.img_paths)
    
        random.shuffle(self.img_paths)
        

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        rgb_img_path,thermal_img_path = self.img_paths[index]
        img_RGB = load_RGB_or_Thermal(rgb_img_path)
        img_Thermal = load_RGB_or_Thermal(thermal_img_path)
        
        img_name = os.path.basename(rgb_img_path)
        gt_path = os.path.join(self.gt_dir, os.path.splitext(img_name)[0] + '.h5')
        target = load_Target(gt_path)
        
        return self.train_transform(img_RGB, img_Thermal, target)
    
    def train_transform(self, RGB, T, keypoints):
        ht, wd, _ = RGB.shape
        st_size = 1.0 * min(wd, ht)
        assert st_size >= self.c_size
        assert len(keypoints) > 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        RGB = RGB[i:i+h, j:j+w, :]
        T = T[i:i+h, j:j+w, :]
        keypoints = keypoints - [j, i]
        idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                   (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
        keypoints = keypoints[idx_mask]

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        gt_discrete = np.expand_dims(gt_discrete, 0)

        RGB = self.RGB_transform(RGB)
        T = self.T_transform(T)
        input = [RGB, T]

        return input, torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float(), st_size