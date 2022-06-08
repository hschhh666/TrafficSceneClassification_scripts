# 根据额外标注的文件生成真值标签文件video_gt.npy
# 注意，假设额外标注了xxx.mp4视频，本程序首先读取xxx.mp4的原始标注文件allGT/xxx.npy，修改后保存到tar_label_folder/xxx.npy下
import numpy as np
import cv2
import os

# 额外标注的文件路径
txt = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/activeLearningDataset/annotate.txt'

# 指定文件路径，一般不用改
origin_label_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/GT/allGT'
tar_label_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/GT/activeLearningAnnotateGT'

# 读取补充标注的标签文件
f = open(txt,'r')
lines = f.readlines()
f.close()

class_name_idx_map = {'shadow':4,'traffic':5,'tunnel':6}
video_class_s_e_map = {}


classname = ''
class_idx = 0
for line in lines:
    if line == '' or line  == '\n':continue
    if line.startswith('#'):
        classname = line[1:-1]
    else:
        v,s,e=line.split()
        s = int(s)
        e = int(e)
        if v not in video_class_s_e_map.keys():
            video_class_s_e_map[v] = {}
        if classname not in video_class_s_e_map[v].keys():
            video_class_s_e_map[v][classname] = []
        video_class_s_e_map[v][classname].append([s,e])


for video_name in video_class_s_e_map.keys():
    origin_label_path = os.path.join(origin_label_folder,video_name+'_gt.npy')
    tar_label_path = os.path.join(tar_label_folder,video_name+'_gt.npy')
    label = np.load(origin_label_path)
    tmp_dict = video_class_s_e_map[video_name]
    for t in tmp_dict.keys():
        c = class_name_idx_map[t] # 类别的索引
        for s_e in tmp_dict[t]:
            s = s_e[0]
            e = s_e[1]
            label[s:e] = c
    
    np.save(tar_label_path, label)

print('Done.')