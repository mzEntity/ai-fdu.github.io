import numpy as np
import os
from glob import glob
import cv2
import json


def generate_data(rgb_path):
    t_path = rgb_path.replace('RGB', 'T')
    rgb = cv2.imread(rgb_path)[..., ::-1].copy()
    t = cv2.imread(t_path)[..., ::-1].copy()
    
    im_h, im_w, _ = rgb.shape
    
    label_path = rgb_path.replace("_RGB.jpg", "_GT.npy")
    
    if not os.path.exists(label_path):
        print(f"{label_path} not exists")
        return rgb, t, None
    
    with open(label_path, 'r') as f:
        label_file = json.load(f)
    points = np.asarray(label_file['points'])
    # print('points', points.shape)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    print(f"{label_path} exists")
    return rgb, t, points


if __name__ == '__main__':

    root_path = '../../my_dataset/'  # dataset root path
    save_dir = '../../rgbtcc_fdu/'

    for phase in ['train', 'val', 'test']:
        sub_dir = os.path.join(root_path, phase)
        sub_save_dir = os.path.join(save_dir, phase)
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)
        rgb_list = glob(os.path.join(sub_dir, '*_RGB.jpg'))
        for rgb_path in rgb_list:
            rgb_name = os.path.basename(rgb_path)
            rgb, t, points = generate_data(rgb_path)
            rgb_save_path = os.path.join(sub_save_dir, rgb_name)
            t_save_path = rgb_save_path.replace('RGB', 'T')
            cv2.imwrite(rgb_save_path, rgb)
            cv2.imwrite(t_save_path, t)
            if points is not None:
                gd_save_path = rgb_save_path.replace('_RGB.jpg', '_GT.npy')
                np.save(gd_save_path, points)
