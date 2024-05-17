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
# from model import CSRNet
from nets.RGBTCCNet import ThermalRGBNet

def load_RGB_or_Thermal(img_path):
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    return img

def load_Target(gt_path):
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    target = cv2.resize(
        target, (target.shape[1]//8, target.shape[0]//8), interpolation=cv2.INTER_CUBIC)*64
    return target

def save_checkpoint(state, is_best, task_id, filename='checkpoint.pth.tar', save_dir='./model/'):  # 添加保存目录参数
    checkpoint_path = os.path.join(save_dir, task_id + filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_model_path = os.path.join(
            save_dir, task_id + 'model_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_model_path)
    
class ImgDataset(Dataset):
    def __init__(self, img_dir, gt_dir, shape=None, shuffle=True, transform=None, batch_size=1, num_workers=4):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.shape = shape
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.rgb_img_paths = [os.path.join(img_dir, filename) for filename in os.listdir(
            img_dir) if filename.endswith('.jpg')]
        
        # 转换为对应的 Thermal 图像路径
        self.thermal_img_paths = [
            path.replace('rgb/', 'tir/').replace('.jpg', 'R.jpg')
            for path in self.rgb_img_paths
        ]

        self.img_paths = list(zip(self.rgb_img_paths, self.thermal_img_paths))

        if shuffle:
            random.shuffle(self.img_paths)

        self.nSamples = len(self.img_paths)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        rgb_img_path,thermal_img_path = self.img_paths[index]
        img_RGB = load_RGB_or_Thermal(rgb_img_path)
        img_Thermal = load_RGB_or_Thermal(thermal_img_path)
        # can optimize
        img_name = os.path.basename(rgb_img_path)
        gt_path = os.path.join(self.gt_dir, os.path.splitext(img_name)[0] + '.h5')
        target = load_Target(gt_path).resize((224, 224))

        if self.transform is not None:
            img_RGB = self.transform(img_RGB)
            img_Thermal = self.transform(img_Thermal)    

        return [img_RGB, img_Thermal], target


lr = 1e-7
original_lr = lr
batch_size = 1
momentum = 0.95
decay = 5*1e-4
epochs = 400
steps = [-1, 1, 100, 150]
scales = [1, 1, 1, 1]
workers = 4
seed = time.time()
print_freq = 30
img_dir = "./dataset/train/rgb/"
gt_dir = "./dataset/train/hdf5s/"
pre = None
task = ""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)
set_seed(42)
    
def main():
    start_epoch = 0
    best_prec1 = 1e6

    model = ThermalRGBNet()

    model = model.to(device)

    criterion = nn.MSELoss(size_average=False).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=decay)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    dataset = ImgDataset(img_dir, gt_dir, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    if pre:
        if os.path.isfile(pre):
            print("=> loading checkpoint '{}'".format(pre))
            checkpoint = torch.load(pre)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(pre))

    for epoch in range(start_epoch, epochs):

        adjust_learning_rate(optimizer, epoch)

        train(model, criterion, optimizer, epoch, train_loader)
        prec1 = validate(model, val_loader)

        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, task)


def train(model, criterion, optimizer, epoch, train_loader):

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    print('epoch %d, processed %d samples, lr %.10f' %
          (epoch, epoch * len(train_loader.dataset), lr))

    model.train()
    end = time.time()

    for i, (img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        img[0] = img[0].to(device)
        img[1] = img[1].to(device)
        print(f"img0: {img[0].shape}, img1: {img[1].shape}")
        count, output, output_normed = model(img)
        target = target.type(torch.FloatTensor).unsqueeze(1).to(device)
        loss = criterion(output, target)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))


def validate(model, val_loader):
    print('begin test')

    model.eval()
    mae = 0

    for i, (img, target) in enumerate(val_loader):
        img[0] = img[0].to(device)
        img[1] = img[1].to(device)
        
        count, output, output_normed = model(img)

        mae += abs(output.data.sum() -
                   target.sum().type(torch.FloatTensor).to(device))

    mae = mae/len(val_loader)
    print(' * MAE {mae:.3f} '
          .format(mae=mae))

    return mae


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    lr = original_lr

    for i in range(len(steps)):

        scale = scales[i] if i < len(scales) else 1

        if epoch >= steps[i]:
            lr = lr * scale
            if epoch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
