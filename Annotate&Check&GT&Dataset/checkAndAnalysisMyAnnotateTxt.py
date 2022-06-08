# 检查我标注的文件是否有格式错误，并分析标注结果
import numpy as np
import os
import cv2
src_file = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/activeLearningDataset/annotate.txt'

# 读取补充标注的标签文件
f = open(src_file,'r')
lines = f.readlines()
f.close()


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

# 数据结构：对于某个类别，各个视频中它所在的位置
class_video_s_e_map = {}
for v, cs in video_class_s_e_map.items():
    for c in cs.keys():
        if c not in class_video_s_e_map.keys():
            class_video_s_e_map[c] = {}
        if v not in class_video_s_e_map[c].keys():
            class_video_s_e_map[c][v] = []
        class_video_s_e_map[c][v] = video_class_s_e_map[v][c]

# 检查标注是否有错误。给定某视频，检查对各类别标注的区间段是否有覆盖。方法是把对该视频的标注区间们按照起点帧号排序，检查一个区间内终点帧号是否大于起点帧号，下个标注区间的起点帧号是否大于上个区间的终点帧号
for v , cs in video_class_s_e_map.items():
    rs = []
    for c in cs.keys():
        rs += video_class_s_e_map[v][c]
    rs.sort(key = lambda x:x[0])
    lst = 0
    for r in rs:
        assert(r[0] > lst)
        assert(r[1] > r[0])
        lst = r[1]

print('Check done, no problem.')


# 检查完毕，开始分析标注结果
classname = ['shadow', 'traffic', 'tunnel']
print('video', end = ' ')
for i in classname:
    print(i, end=' ')
print()
for v, cs in video_class_s_e_map.items():
    print(v, end = ' ')
    for c in classname:
        if c not in cs.keys():
            print(0, end = ' ')
        else:
            rs = cs[c]
            cnt = 0
            for r in rs:
                cnt += r[1] - r[0]
            print(cnt, end = ' ')
    print()

        

print('Done.')
