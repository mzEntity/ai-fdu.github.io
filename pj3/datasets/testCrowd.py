import PIL.Image as Image
import torchvision.transforms.functional as F
import torch
# from model import CSRNet
from nets.RGBTCCNet import ThermalRGBNet
from torchvision import transforms

import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def load_RGB_or_Thermal(img_path):
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    return img

class TestCrowd(Dataset):
    def __init__(self, batch_size=1):
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
        
        # 形成RGB、T的路径对，放在self.img_paths中，rgb/4.jpg对应tir/4R.jpg
        self.img_dir = './dataset/test/rgb/'

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

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        rgb_img_path, thermal_img_path = self.img_paths[index]
        img_RGB = load_RGB_or_Thermal(rgb_img_path)
        img_Thermal = load_RGB_or_Thermal(thermal_img_path)

        img_RGB = self.RGB_transform(img_RGB)
        img_Thermal = self.T_transform(img_Thermal)    
        
        # 获得index，比如10.jpg，获得10
        parts = rgb_img_path.split('/')
        filename = parts[-1]
        file_number = filename.split('.')[0]

        return [img_RGB, img_Thermal, int(file_number)]