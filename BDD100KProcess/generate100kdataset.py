from tqdm import tqdm
import numpy as np
import json
import os
import shutil

split = 'train'

for split in ['train','val']:

    json_file = 'E:/Research/2021TrafficSceneClassification/Datasets/BDD100K/bdd100k_det_20_labels_trainval/bdd100k/labels/det_20/det_%s.json'%split
    src_img_path = 'E:/Research/2021TrafficSceneClassification/Datasets/BDD100K/bdd100k_images_100k/bdd100k/images/100k/%s'%split
    tar_img_path = 'C:/Users/A/Desktop/mydataset/%s'%split


    # 读取json文件
    json_file = open(json_file,'r')
    jsondata = json.load(json_file)
    json_file.close()

    class_img_dict = {}

    pbar = tqdm(jsondata)
    for item in pbar:
        img_name = item['name']
        scene = item['attributes']['scene']
        if scene not in class_img_dict.keys():
            class_img_dict[scene] = []
        class_img_dict[scene].append(img_name)

    print('%s data number:'%split)
    for k in class_img_dict.keys():
        print(k, len(class_img_dict[k]))


    for k in class_img_dict.keys():
        img_list = class_img_dict[k]
        img_list = [os.path.join(src_img_path,i) for i in img_list]
        tar_class_path = os.path.join(tar_img_path, k)
        if not os.path.exists(tar_class_path):
            os.makedirs(tar_class_path)
        pbar = tqdm(img_list,desc=k)
        for src_img in pbar:
            shutil.copy(src_img, tar_class_path)
