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

from datasets.testCrowd import TestCrowd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 1
test_dataset = TestCrowd(batch_size=batch_size)
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
