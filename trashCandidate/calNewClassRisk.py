# 对于标注新类别的视频，计算新类别的risk是多少，以此反映新类别和risk的相关性
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# 要处理的视频名
video_name = '20180404_134505_2018-04-04'

# 新类别标注文件
src_file = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/activeLearningAnnotate/annotate.txt'

# 指定文件夹路径
deep_learning_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/experimentRes/Contras/videoFeatTrainedWithDatasetA'

# 读取补充标注的标签文件
f = open(src_file,'r')
lines = f.readlines()
f.close()

# 读取risk、pred、feat的npy文件
risk = os.path.join(deep_learning_folder, video_name+'_risks.npy')
pred = os.path.join(deep_learning_folder, video_name+'_predLabels.npy')
feat = os.path.join(deep_learning_folder, video_name+'_memoryFeature.npy')
risk = np.load(risk)
pred = np.load(pred)
feat = np.load(feat)
feat = feat / np.linalg.norm(feat,axis=1, keepdims=True) # 撒币了，在计算特征的时候忘了归一化了


# 每帧的risk
min_risk = np.min(risk, axis=1)

# 数据结构：每个视频中，各个类别所在的位置
video_class_s_e_map = {}
for line in lines:
    if line == '' or line.isspace():continue
    if line.startswith('#'):
        classname = line.strip()[1:]
    else:
        v,s,e=line.split()
        s = int(s)
        e = int(e)
        assert(s<e) # 检查终点帧号是否大于起点帧号
        if v not in video_class_s_e_map.keys():
            video_class_s_e_map[v] = {}
        if classname not in video_class_s_e_map[v].keys():
            video_class_s_e_map[v][classname] = []
        video_class_s_e_map[v][classname].append([s,e])

classname = ['shadow', 'traffic', 'tunnel']
cs = video_class_s_e_map[video_name]
fno = np.shape(risk)[0]
for c in classname:
    if c not in cs.keys():continue
    mask = np.zeros(fno, dtype=bool) # 如果mask[i]=True，表示位置i的标签是c。也就是说mask标记了视频v中标签为c的索引
    rs = cs[c]
    for r in rs:
        mask[r[0]:r[1]] = True
    selected_frame_risk = min_risk[mask]
    plt.figure()
    plt.scatter(list(range(np.shape(selected_frame_risk)[0])), selected_frame_risk, s = 1)
    plt.title(c)
plt.show()