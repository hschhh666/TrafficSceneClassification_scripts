#  分析数据集
# 统计abnormal时长
# 统计new和hard占abnormal的比例
# 统计new和hard中各类别的数量
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# 指定文件夹路径
origin_gt_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/GT/allGT'
relabeled_gt_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/GT/activeLearningAnnotateGT'
deep_learning_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/experimentRes/Contras/videoFeatTrainedWithDatasetA'
# excel文件路径，excel记录了数据集划分，即某个视频是B数据集还是C数据集
excel_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/datasetAnalysis.xlsx'

# 读取excel，判断某视频得到每个视频的是数据集B还是C
excel_file = pd.read_excel(excel_path,sheet_name='ActiveLearningDataset')
video_names_in_excel = excel_file.loc[:,'Video'].values # excel 中所有视频名
video_dataset_split = excel_file.loc[:,'Re-split'].values # 某视频是A还是B还是C
video_fpss = excel_file.loc[:,'FPS'].values

# 类别名与索引的对应关系
class_name_idx_map = {'Highway':0, 'Local':1, 'Ramp':2, 'Urban':3}

# 统计每个视频中new/hard的时长
video_abnormalScene_class_dict = {}

# 每个类别的数量
class_count_dict = {'Highway':0,'Local':0,'Ramp':0, 'Urban':0}


# 逐个访问视频
for excel_row_idx, video_name in enumerate(video_names_in_excel):
    dataset_type = video_dataset_split[excel_row_idx]
    if dataset_type != 'B': continue
    fps = video_fpss[excel_row_idx]

    video_abnormalScene_class_dict[video_name] = {}
    video_abnormalScene_class_dict[video_name]['new']  = {}
    video_abnormalScene_class_dict[video_name]['hard'] = {}

    for key in video_abnormalScene_class_dict[video_name].keys():
        for k in class_name_idx_map.keys():
            video_abnormalScene_class_dict[video_name][key][k] = 0

    origin_gt = np.load(os.path.join(origin_gt_folder,video_name+'_gt.npy'))
    relabeld_gt = origin_gt

    if os.path.exists(os.path.join(relabeled_gt_folder,video_name+'_gt.npy')):
        relabeld_gt = np.load(os.path.join(relabeled_gt_folder,video_name+'_gt.npy'))
    
    # 读取文件
    risk = os.path.join(deep_learning_folder, video_name+'_risks.npy')
    pred = os.path.join(deep_learning_folder, video_name+'_predLabels.npy')
    risk = np.load(risk)
    pred = np.load(pred)

    # 视频总帧数
    frame_num = np.shape(origin_gt)[0]
    # 每帧的risk
    min_risk = np.min(risk, axis=1)
    # 计算一段时间窗内最小risk的均值方差
    time_window_size = 150 # 单位：帧
    time_window_risk_mean = np.zeros_like(min_risk)

    # 计算一定时间范围内的平均risk
    for i in range(frame_num):
        if i-time_window_size < 0 or i+time_window_size > frame_num: continue
        tmp_risk = min_risk[i-time_window_size:i+time_window_size]
        time_window_risk_mean[i] = np.mean(tmp_risk)

    abnormal_idx = set(np.where(time_window_risk_mean > 0.03)[0])
    newScene_idx = set(np.where(relabeld_gt >= 4)[0])
    newScene_idx = abnormal_idx.intersection(newScene_idx)
    hardScene_idx = abnormal_idx - newScene_idx
    newScene_idx = set(np.where(relabeld_gt >= 4)[0])
    per_class_idx = []
    for k,i in class_name_idx_map.items():
        tmp = set(np.where(origin_gt == i)[0])

        per_class_idx.append(tmp)
        class_count_dict[k] += (len(tmp)/fps)

    for c in class_name_idx_map.keys():
        i = class_name_idx_map[c]
        tmp = len(newScene_idx.intersection(per_class_idx[i]))/fps
        video_abnormalScene_class_dict[video_name]['new'][c] = tmp
        tmp = len(hardScene_idx.intersection(per_class_idx[i]))/fps
        video_abnormalScene_class_dict[video_name]['hard'][c] = tmp

# 统计new和hard在abnormal中的数量
abnormal_contex = {'new':0,'hard':0}
for v in video_abnormalScene_class_dict.keys():
    for t in video_abnormalScene_class_dict[v]:
        tmp = 0
        for c in video_abnormalScene_class_dict[v][t]:
            tmp += video_abnormalScene_class_dict[v][t][c]
        abnormal_contex[t] += tmp

print(abnormal_contex)

# 统计各类别在new中的数量
new_contex = {'Highway':0,'Local':0,'Ramp':0, 'Urban':0}
for v in video_abnormalScene_class_dict.keys():
    for c in video_abnormalScene_class_dict[v]['new']:
        tmp = video_abnormalScene_class_dict[v]['new'][c]
        new_contex[c] += tmp

print(new_contex)

# 统计各类别在hard中的数量
hard_contex = {'Highway':0,'Local':0,'Ramp':0, 'Urban':0}
for v in video_abnormalScene_class_dict.keys():
    for c in video_abnormalScene_class_dict[v]['hard']:
        tmp = video_abnormalScene_class_dict[v]['hard'][c]
        hard_contex[c] += tmp

print(hard_contex)