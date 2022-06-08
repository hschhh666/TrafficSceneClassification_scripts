# 合并单个视频数据到完整train、val数据集
import numpy as np
import cv2
import os
import pandas as pd
from tqdm import tqdm
import time
import shutil
from utiles import label_conventer, hms2s

# =====================================输入参数区=====================================
tar_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/dataset'

single_video_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/singleVideo'

# 文件路径设置
excel_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/datasetAnalysis.xlsx'
excel_file = pd.read_excel(excel_path,sheet_name='20220327DatasetGeneration')
video_names = excel_file.loc[:,'Video'].values # 所有视频文件名
train_or_val = excel_file.loc[:,'Re-split'].values # 所有视频文件名


label_name = ['Highway','Local','Ramp','Urban']
for i in label_name:
    os.makedirs(os.path.join(tar_path,'train',i))
    os.makedirs(os.path.join(tar_path,'val',i))

for i, video_name in enumerate(video_names):
    print('[%d/%d]'%(i+1, len(video_names)))
    t_or_v = train_or_val[i]
    src_root_path = os.path.join(single_video_path,video_name)
    for l in label_name:
        src_class_path = os.path.join(src_root_path, l)
        tar_class_path = os.path.join(tar_path,t_or_v,l)
        for root,_,files in os.walk(src_class_path):
            files = [os.path.join(root,f) for f in files]
        for f in files:
            shutil.copy(f,tar_class_path)
