import os
import glob
import cv2

src_path = "../pj3/dataset/"
dst_path = "../../my_dataset/"



print("start move jpgs")

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
        Img_data = cv2.imread(rgb_path)
        
        tir_path = rgb_path.replace("rgb", "tir").replace(".jpg", "R.jpg")

        rgb = cv2.imread(rgb_path)[..., ::-1].copy()
        t = cv2.imread(tir_path)[..., ::-1].copy()
        
        rgb_save_path = os.path.join(dst_base_path, os.path.basename(rgb_path)).replace(".jpg", "_RGB.jpg")
        t_save_path = rgb_save_path.replace("_RGB.jpg", "_T.jpg")

        cv2.imwrite(rgb_save_path, rgb)
        cv2.imwrite(t_save_path, t)
        
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
    json_path = os.path.join(dst_base_path, os.path.basename(label_path)).replace("R.xml", "_GT.json")
    points = parse_xml(label_path)
    print(f"parse {i}: {len(points)}")
    save_to_json(points, json_path)
    i = i + 1

import shutil    
    
src_base_path = os.path.join(dst_path, "train/")
dst_base_path = os.path.join(dst_path, "val/")
if not os.path.exists(dst_base_path):
    os.makedirs(dst_base_path)

i = 0
for json_path in glob.glob(os.path.join(src_base_path, '*_GT.json')):
    if i % 10 != 0:
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
    
    
