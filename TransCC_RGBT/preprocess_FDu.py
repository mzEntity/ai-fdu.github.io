import os
import glob
import cv2
import numpy as np

src_path = "../../my_dataset/"
dst_path = "./my_dataset/"

def resizeJPG(image):
    height, width = image.shape[:2]

    resized_image = cv2.resize(image, (width // 2, height // 2))
    
    output_image = np.zeros((height, width, 3), dtype=np.uint8)

    output_image[0:height//2, 0:width//2] = resized_image
    output_image[0:height//2, width//2:width] = resized_image
    output_image[height//2:height, 0:width//2] = resized_image
    output_image[height//2:height, width//2:width] = resized_image
    return output_image


print("start move jpgs")

more_jpg = True
skip_big = True

options = ["train/", "test/"]
option2 = ""

for option in options:
    src_base_path = os.path.join(src_path, option)
    dst_base_path = os.path.join(dst_path, option)

    if not os.path.exists(dst_base_path):
        os.makedirs(dst_base_path)

    rgb_base_path = os.path.join(src_base_path, "rgb/")
    tir_base_path = os.path.join(src_base_path, "tir/")

    rgb_paths = []
    for rgb_path in glob.glob(os.path.join(rgb_base_path, '*.jpg')):
        rgb_paths.append(rgb_path)
        
    i = 0
    for rgb_path in rgb_paths:
        print(f"{option}: {i}")
        
        if option == "train/" and skip_big:
            index = int(os.path.basename(rgb_path).split('.')[0])
            if index < 124 or (index > 200 and index < 411) or (index > 708 and index < 1389) or (index > 1733 and index < 2000):
                print(f"skip {index}")
                continue
        
        Img_data = cv2.imread(rgb_path)
        
        tir_path = rgb_path.replace("rgb", "tir").replace(".jpg", "R.jpg")

        rgb = cv2.imread(rgb_path)
        t = cv2.imread(tir_path)
        
        rgb_save_path = os.path.join(dst_base_path, os.path.basename(rgb_path)).replace(".jpg", "_RGB.jpg")
        t_save_path = rgb_save_path.replace("_RGB.jpg", "_T.jpg")

        cv2.imwrite(rgb_save_path, rgb)
        cv2.imwrite(t_save_path, t)
        
        if option == "train/" and more_jpg:
            new_rgb = resizeJPG(rgb)
            new_t = resizeJPG(t)
            new_rgb_save_path = rgb_save_path.replace("_RGB.jpg", "10000_RGB.jpg")
            new_t_save_path = t_save_path.replace("_T.jpg", "10000_T.jpg")
            cv2.imwrite(new_rgb_save_path, new_rgb)
            cv2.imwrite(new_t_save_path, new_t)
        
        i = i + 1
        
        
import xml.etree.ElementTree as ET
import json

def parse_xml(xml_path):
    point_list = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall('object'):
        try:
            x = int(obj.find('point/x').text)
            y = int(obj.find('point/y').text)
        except:
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)
            x = (xmin + xmax) // 2
            y = (ymin + ymax) // 2
        point_list.append([x, y])
    return point_list


def save_to_json(points, json_path):
    data = {
        "points": points,
        "count": len(points)
    }
    
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

print("start parse xmls")        
        
src_base_path = os.path.join(src_path, "train/")
dst_base_path = os.path.join(dst_path, "train/")

if not os.path.exists(dst_base_path):
    os.makedirs(dst_base_path)

label_base_path = os.path.join(src_base_path, "labels/")

i = 0
for label_path in glob.glob(os.path.join(label_base_path, '*R.xml')):
    if skip_big:
        index = int(os.path.basename(label_path).split('R')[0])
        if index < 124 or (index > 200 and index < 411) or (index > 708 and index < 1389) or (index > 1733 and index < 2000):
            print(f"skip {index}")
            continue
        
    json_path = os.path.join(dst_base_path, os.path.basename(label_path)).replace("R.xml", "_GT.json")
    rgb_path = json_path.replace("_GT.json", "_RGB.jpg")
    rgb_data = cv2.imread(rgb_path)
    height, width = rgb_data.shape[:2]
    
    points = parse_xml(label_path)
    print(f"parse {i}: {len(points)}")
    save_to_json(points, json_path)
    if more_jpg:
        new_points = []
        new_json_path = json_path.replace("_GT.json", "10000_GT.json")
        for point in points:
            new_points.append([(point[0] + width) / 2, (point[1] + height) / 2])
            new_points.append([(point[0] + width) / 2, point[1]])
            new_points.append([point[0], (point[1] + height) / 2])
            new_points.append([point[0], point[1]])
        save_to_json(new_points, new_json_path)
    i = i + 1

import shutil    
    
src_base_path = os.path.join(dst_path, "train/")
dst_base_path = os.path.join(dst_path, "val/")
if not os.path.exists(dst_base_path):
    os.makedirs(dst_base_path)

i = 0
for json_path in glob.glob(os.path.join(src_base_path, '*_GT.json')):
    if i % 10 != 0:
        i = i + 1
        continue
    rgb_path = json_path.replace("_GT.json", "_RGB.jpg")
    t_path = json_path.replace("_GT.json", "_T.jpg")
    
    rgb_dst_path = os.path.join(dst_base_path, os.path.basename(rgb_path))
    t_dst_path = os.path.join(dst_base_path, os.path.basename(t_path))
    json_dst_path = os.path.join(dst_base_path, os.path.basename(json_path))
    
    print(f"move to val {i}")
    shutil.move(rgb_path, rgb_dst_path)
    shutil.move(t_path, t_dst_path)
    shutil.move(json_path, json_dst_path)
    i = i + 1
    
    
