import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from torch import dtype
from tqdm import tqdm
from utiles import *



# ==========================输入参数==========================
risk_img_h = 200
time_range = 10

# ==========================文件路径，不用改==========================
data_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/val_center_norm'


f = []
for _,_,tmpf in os.walk(data_path):
    for i in tmpf:
        if i[-3:] == 'mp4':
            f.append(i[:-4])

for video_name in f:
    risk_name = video_name+'_to_classes_risk'
    if os.path.exists(os.path.join(data_path, risk_name + '.npy')) and not os.path.exists(os.path.join(data_path,video_name+'_withRisk.avi')):


        # ==========================加载数据==========================
        all_risk = np.load(os.path.join(data_path, risk_name + '.npy'))
        cap = cv2.VideoCapture(os.path.join(data_path,video_name+'.mp4'))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.join(data_path,video_name+'_withRisk.avi'), fourcc, fps, (width, height + risk_img_h))




        risk_img_w = width
        last_risk_img = 255 * np.ones((risk_img_h,risk_img_w,3), dtype=np.uint8)
        colors = [[0,0,255],[71,173,112],[162,72,149],[240,176,2]]

        double_clip_width = ((1/fps) * risk_img_w / (time_range*2)) # 每帧risk图平移的量
        left_clip_witdh = 0

        # for i in range(1000):
        #     all_risk[i,0] = i%2


        last_xy = np.zeros((4,2),dtype=int)
        for i in range(time_range * fps):
            cur_time = i/fps # 单位 秒
            values = all_risk[i,:]
            for j in range(4):
                value = values[j]
                x,y = pos_converter(risk_img_h, risk_img_w, time_range, cur_time, value)
                if i == 0:
                    last_xy[j] = [x,y]
                else:
                    cv2.line(last_risk_img, (last_xy[j,0],last_xy[j,1]), (x,y), colors[j],thickness=2)
                    last_xy[j] = [x,y]

        pbar = tqdm(range(frame_count),desc=video_name)
        for i in pbar:
            ret,img = cap.read()

            should_clip = double_clip_width + left_clip_witdh
            cur_clip = int(should_clip)
            left_clip_witdh = should_clip - cur_clip

            for j in range(4):
                last_xy[j,0] = last_xy[j,0]-cur_clip
            
            cur_risk_img = last_risk_img[:,cur_clip:,:]
            empty = 255 * np.ones((risk_img_h, cur_clip, 3), dtype=np.uint8) # 每帧risk新加的空白图
            cur_risk_img = np.concatenate((cur_risk_img, empty), axis=1)
            if i + time_range * fps < frame_count:
                values = all_risk[i + time_range * fps, :]
                for j in range(4):
                    value = values[j]
                    x,y = pos_converter(risk_img_h, risk_img_w, time_range, time_range, value)
                    cv2.line(cur_risk_img, (last_xy[j,0],last_xy[j,1]), (x,y), colors[j], thickness=2)
                    last_xy[j] = [x,y]
                    
            
            last_risk_img = cur_risk_img
            

            if not ret:
                continue
            cv2.putText(img,str(i),(30,50),2,1,(0,0,255),2)
            res_img = np.concatenate((cur_risk_img, img), axis=0)
            cur_risk_pos = int(risk_img_w/2)
            cv2.line(res_img, (cur_risk_pos, 0),(cur_risk_pos, risk_img_h), color=(0,0,0))
            # cv2.imshow('img', res_img)
            # cv2.waitKey(1)


            out.write(res_img)

        out.release()
        cap.release()
