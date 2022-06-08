# 合并单个视频数据到完整train、val数据集
import numpy as np
import cv2
import os
import pandas as pd
from tqdm import tqdm
import time
import shutil
from utiles import label_conventer, hms2s
import json


tar_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/Ramp'
# 文件路径设置
excel_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/datasetAnalysis.xlsx'
excel_file = pd.read_excel(excel_path,sheet_name='重新划分数据集')
video_names = excel_file.loc[:,'Video'].values # 所有视频文件名
train_or_val = excel_file.loc[:,'Re-split'].values # 所有视频文件名

os.makedirs(os.path.join(tar_path,'train','Ramp'))
os.makedirs(os.path.join(tar_path,'val','Ramp'))


start_time = time.time()
for hhh, video_name in enumerate(video_names):
    print('%s[%d/%d] %s %s'%('='*30,hhh+1, len(video_names), video_name,'='*30))
    
    t_or_v = train_or_val[hhh] # 当前是训练集还是测试集
    t_or_v = 'train' if (t_or_v != 'val' and t_or_v != 'train')  else t_or_v

    video_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/FINALDATA/VIDS'
    json_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/FINALDATA/JSON'

    video_file = os.path.join(video_path, video_name+'.mp4')
    json_file = os.path.join(json_path, video_name+'.json')

    # 读取视频
    cap = cv2.VideoCapture(video_file)
    fno = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 读取json文件
    json_file = open(json_file,'r')
    jsondata = json.load(json_file)
    json_file.close()

    # 开始解析json文件
    vid_annot_list = []
    for item in jsondata['data']:
        class_name = item['category']
        attributes = item['attributes']
        starttime = attributes[0]['value']
        endtime = attributes[1]['value']
        tmp = [starttime, endtime]
        if class_name == 'Highway':
            tmp.append('Highway')
        if class_name == 'Non Highway':
            for l in attributes:
                if l['name'] == 'Road Environment Type':
                    tmp.append(l['value'])
            
        if len(tmp) == 3:
            vid_annot_list.append(tmp)
    
    vid_annot_list.sort(key=lambda x: [x[0],x[1]])

    for vid_annot in vid_annot_list:
        start = vid_annot[0]
        end = vid_annot[1]
        clas = vid_annot[2]
        if clas != 'Ramp': continue
        start = hms2s(start) * fps
        end = hms2s(end) * fps

        
        if t_or_v == 'train':
            interval_sec = 3 # 间隔这么多秒采样
            discard = 0.5 # 丢弃开始和结束的百分之这么多
        elif t_or_v == 'val':
            interval_sec = 0.4 # 间隔这么多秒采样
            discard = 0.2 # 丢弃开始和结束的百分之这么多
        else:
            raise ValueError('Error!')
        
        s = start
        e = end
        
        start = start + int((e-s)*discard)
        end = end - int((e-s)*discard)
        
        interval_frame = int(interval_sec * fps)
        sample_idx = list(range(start,end, interval_frame))

        for i in sample_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret,frame = cap.read()
            if not ret:
                print('Read frame error in %s frame %d'%(video_name,i))
            frame = cv2.resize(frame,(400,224))
            frame_name = video_name + '-' + str(i).rjust(10,'0')+'.png'
            frame_name = os.path.join(tar_path,t_or_v,'Ramp', frame_name)
            cv2.imwrite(frame_name, frame)

    cap.release()