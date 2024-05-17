import PIL.Image as Image
import torchvision.transforms.functional as F
import torch
# from model import CSRNet
from .nets.RGBTCCNet import ThermalRGBNet
from torchvision import transforms

import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def load_RGB_or_Thermal(img_path):
    img = Image.open(img_path).convert('RGB')
    return img

class ImgDataset(Dataset):
    def __init__(self, img_dir, batch_size=1):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                0.229, 0.224, 0.225]),
        ])
        self.img_dir = img_dir
        self.batch_size = batch_size

        self.rgb_img_paths = [os.path.join(img_dir, filename) for filename in os.listdir(
            img_dir) if filename.endswith('.jpg')]
        
        # 转换为对应的 Thermal 图像路径
        self.thermal_img_paths = [
            path.replace('rgb/', 'tir/').replace('.jpg', 'R.jpg')
            for path in self.rgb_img_paths
        ]

        self.img_paths = list(zip(self.rgb_img_paths, self.thermal_img_paths))

        self.nSamples = len(self.img_paths)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        rgb_img_path,thermal_img_path = self.img_paths[index]
        img_RGB = load_RGB_or_Thermal(rgb_img_path)
        img_Thermal = load_RGB_or_Thermal(thermal_img_path)

        if self.transform is not None:
            img_RGB = self.transform(img_RGB)
            img_Thermal = self.transform(img_Thermal)    
        
        parts = rgb_img_path.split('/')
        # 获取文件名部分（即最后一部分）
        filename = parts[-1]
        # 再次使用split()分割以去掉扩展名
        file_number = filename.split('.')[0]

        return [img_RGB, img_Thermal, int(file_number)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_dir = './dataset/test/rgb/'
batch_size = 1
test_dataset = ImgDataset(img_dir=img_dir,batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = ThermalRGBNet()
model = model.to(device)
checkpoint = torch.load('./model/model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

# 用于存储结果的列表
results = []

model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 关闭梯度计算，节省内存和计算资源
    for i, (img_RGB, img_Thermal,file_number) in enumerate(test_loader):
        print(i)
        if i > 2:
            break
        img_RGB = img_RGB.to(device)
        img_Thermal = img_Thermal.to(device)
        file_number = int(file_number)
        count, output, output_normed = model([img_RGB, img_Thermal]) 
        ans = output.detach().cpu().sum()  
        formatted_ans = "{:.2f}".format(ans.item())
        results.append([file_number, f"{file_number},{formatted_ans}\n"])

results.sort()
# 一次性写入所有结果到文件
with open('./ans.txt', 'w') as file:
    for result in results:
        file.writelines(result[1])
