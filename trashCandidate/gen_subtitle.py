import numpy as np
import os
import cv2
from tqdm import tqdm
import json

json_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/FINALDATA/JSON'
video_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/FINALDATA/VIDS'

json_files = []
for _,_,json_files in os.walk(json_folder):
    pass
json_files = [os.path.join(json_folder,i) for i in json_files]


pbar = tqdm(json_files)
for json_loc in pbar:
    json_file = open(json_loc,'r')
    data = json.load(json_file)
    json_file.close()

    # 开始解析json文件
    vid_annot_list = []
    for item in data['data']:
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
    
    # 开始创建字幕文件
    name = json_loc.split('\\')[-1][:-4]
    name = os.path.join(video_folder, name+'srt')
    
    f = open(name,'w')
    for i, context in enumerate(vid_annot_list):
        start = context[0]
        start += '.000'
        start = '00:'+start
        end = context[1]
        end += '.000'
        end = '00:'+end
        clas = context[2]
        f.write('%d\n'%(i+1))
        f.write('%s --> %s\n'%(start,end))
        f.write('%s\n\n'%(clas))
    f.close()
    pass