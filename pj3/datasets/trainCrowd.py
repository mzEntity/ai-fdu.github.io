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

def load_RGB_or_Thermal(img_path):
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    return img

def load_Target(gt_path):
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    target = cv2.resize(target, (224, 224), interpolation=cv2.INTER_CUBIC)
    return target

class TrainCrowd(Dataset):
    def __init__(self, shape=None, batch_size=1, num_workers=4):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
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

        if self.transform is not None:
            img_RGB = self.transform(img_RGB)
            img_Thermal = self.transform(img_Thermal)    

        return [img_RGB, img_Thermal], target