import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# 指定文件夹路径
origin_gt_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/GT/allGT'
relabeled_gt_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/GT/activeLearningAnnotateGT'
deep_learning_root_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/experimentRes/Contras'
# excel文件路径，excel记录了数据集划分，即某个视频是B数据集还是C数据集
excel_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/datasetAnalysis.xlsx'

# 要处理哪些实验结果
deep_learning_res_name_list = [ 'videoFeatTrainedWithDatasetA','video_feat_A_HS', 'video_feat_A_HS_shadow', 'video_feat_A_HS_shadowTraffic', 'video_feat_A_HS_allNewScene',]


# 读取excel，判断某视频得到每个视频的是数据集B还是C
excel_file = pd.read_excel(excel_path,sheet_name='ActiveLearningDataset')
video_names_in_excel = excel_file.loc[:,'Video'].values # excel 中所有视频名
video_dataset_split = excel_file.loc[:,'Re-split'].values # 某视频是A还是B还是C

# 类别名与索引的对应关系
class_name_idx_map = {'Highway':0, 'Local':1, 'Ramp':2, 'Urban':3, 'shadow':4,'traffic':5,'tunnel':6}

# 每个实验下，每个类别的risk
exp_class_risk_dict = {}
for exp_name in deep_learning_res_name_list:
    exp_class_risk_dict[exp_name] = {}
    for class_name in class_name_idx_map.keys():
        exp_class_risk_dict[exp_name][class_name] = []

# 逐个访问视频
for excel_row_idx, video_name in enumerate(video_names_in_excel):
    dataset_type = video_dataset_split[excel_row_idx]
    if dataset_type != 'C': continue

    # 获取真值，如果有新标注的则读取新标注真值，否则读取原真值
    if os.path.exists(os.path.join(relabeled_gt_folder,video_name+'_gt.npy')):
        gt = np.load(os.path.join(relabeled_gt_folder,video_name+'_gt.npy'))
    else:
        gt = np.load(os.path.join(origin_gt_folder,video_name+'_gt.npy'))

    # 计算每个实验的情况
    for exp_idx, exp_name in enumerate(deep_learning_res_name_list):
        deep_learning_folder = os.path.join(deep_learning_root_folder, exp_name)

        # 读取文件
        risk = os.path.join(deep_learning_folder, video_name+'_risks.npy')
        pred = os.path.join(deep_learning_folder, video_name+'_predLabels.npy')
        feat = os.path.join(deep_learning_folder, video_name+'_memoryFeature.npy')
        risk = np.load(risk)
        pred = np.load(pred)
        feat = np.load(feat)
        feat = feat / np.linalg.norm(feat,axis=1, keepdims=True) # 撒币了，在计算特征的时候忘了归一化了

        # 视频总帧数
        frame_num = np.shape(gt)[0]
        # 每帧的risk
        min_risk = np.min(risk, axis=1)

        for class_name in class_name_idx_map.keys():
            class_idx = class_name_idx_map[class_name]
            tmp_risk = list(min_risk[gt == class_idx])
            exp_class_risk_dict[exp_name][class_name] += tmp_risk

print('Experiment', end = ' ')
for i in class_name_idx_map.keys():
    print(i, end=' ')
print()
for exp_name in deep_learning_res_name_list:
    print(('%s'%exp_name).ljust(30,' '), end = ' ')
    for class_name in class_name_idx_map.keys():
        exp_class_risk_dict[exp_name][class_name] = np.mean(exp_class_risk_dict[exp_name][class_name])
        print('%.3f'%exp_class_risk_dict[exp_name][class_name], end=' ')
    print()


exit()

