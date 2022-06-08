# 读取excel文件确定处理哪些视频，处理单个视频，提取单个视频的数据集
import numpy as np
import cv2
import os
import pandas as pd
from tqdm import tqdm
import time
from utiles import label_conventer, hms2s


# =====================================输入参数区=====================================
tar_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/CoarseData/BalenceSampled/test'

# =====================================开始基本配置=====================================
# 不可变参数区
label_name = ['Highway','Local','Ramp','Urban']
interval_sec = [17.3,9.24,1.9,39.5] # 各个类别间隔采样的秒数 训练集
interval_sec = [14.88,2.76,1,18.89] # 各个类别间隔采样的秒数 测试集


# 文件路径设置
excel_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/datasetAnalysis.xlsx'
excel_file = pd.read_excel(excel_path,sheet_name='Sheet')
video_names = excel_file.loc[:,'Video'].values # 所有视频文件名
train_or_val = excel_file.loc[:,'Split'].values

class_num = [0,0,0,0]

start_time = time.time()
for hhh, video_name in enumerate(video_names):
    t_or_v = train_or_val[hhh]
    if t_or_v != 'val': continue
    print('%s[%d/%d] %s %s'%('='*30,hhh+1, len(video_names), video_name,'='*30))
    
    video_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/FINALDATA/VIDS'
    csv_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/FINALDATA/CSV'
    video_file = os.path.join(video_path, video_name+'.mp4')
    csv_file = os.path.join(csv_path,video_name+'.csv')

    # 读取视频
    cap = cv2.VideoCapture(video_file)
    fno = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 读取标签
    f = open(csv_file,'r')
    csv_file = f.readlines()
    f.close()
    csv_file = csv_file[1:]
    labels = [label_conventer(int(i.split(',')[-2])) for i in csv_file] #得到该视频逐帧的标签

    assert(len(labels) == fno)

    # ===================================== 以上，基础配置结束=====================================

    idxes = list(range(0,fno)) # 每帧的索引
    idxes = np.array(idxes,dtype=int)
    labels = np.array(labels, dtype=int)

    mask = np.ones(fno,dtype=bool)

    masked_labels = labels[mask] # 排除了不合理标签后的图片标签
    masked_idxes = idxes[mask] # 排除了不合理标签后的图片索引

    # 分别采样每类
    for l in range(4):# l 是类别序号
        cur_class_name = label_name[l]
        cur_interval_frame = int(interval_sec[l] * fps)
        cur_class_idx = masked_idxes[masked_labels == l] # 当前类别的所有索引
        sample_idx = np.array(list(range(0,len(cur_class_idx), cur_interval_frame)), dtype=int)
        sample_class_idx = cur_class_idx[sample_idx]
        
        cur_tar_path = os.path.join(tar_path,cur_class_name)
        if os.path.exists(cur_tar_path): 
            print('Folder %s exit. Please confirm.'%(cur_tar_path))
        else:
            os.makedirs(cur_tar_path)
        
        
        sample_class_idx = list(sample_class_idx)
        class_num[l] += len(sample_class_idx)
        pbar = tqdm(sample_class_idx,desc=cur_class_name)
        for i in pbar:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret,frame = cap.read()
            if not ret:
                print('Read frame error in %s frame %d'%(video_name,i))
                continue
            frame = cv2.resize(frame,(400,224))
            frame_name = video_name + '-' + str(i).rjust(10,'0')+'.png'
            frame_name = os.path.join(cur_tar_path, frame_name)
            cv2.imwrite(frame_name, frame)

    print('Using time %s'%time.strftime('%H:%M:%S', time.gmtime(time.time()- start_time)))
    print()
    cap.release()
    pass

total = 0
for i in range(4):
    print('%s: %d'%(label_name[i], class_num[i]))
    total += class_num[i]
print('Total: %d'%total)
