# 根据自己额外标注的文本文件生成数据集
import numpy as np
import os
import cv2

tarpath = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/supplementAnnotateSmaleClass'
txt = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/supplementAnnotateSmaleClass/supplement.txt'
videopath = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/FINALDATA/VIDS'

f = open(txt,'r')
lines = f.readlines()
f.close()

classname = ''
savepath = ''
for line in lines:
    if line == '' or line  == '\n':continue
    if line.startswith('#'):
        classname = line[1:-1]
        savepath = os.path.join(tarpath, classname)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
    else:
        v,s,e=line.split()
        v+='.mp4'
        video_name = v
        v = os.path.join(videopath,v)
        s = int(s)
        e = int(e)
        cap = cv2.VideoCapture(v)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        sampleidx = list(range(s,e,fps*1))
        for i in sampleidx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret,frame = cap.read()
            if not ret:
                print('Read frame error in %s frame %d'%(video_name,i))
                continue
            frame = cv2.resize(frame,(400,224))
            frame_name = video_name + '-' + str(i).rjust(10,'0')+'.png'
            frame_name = os.path.join(savepath, frame_name)
            cv2.imwrite(frame_name, frame)
        cap.release()



        

