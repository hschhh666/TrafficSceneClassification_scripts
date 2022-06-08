import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# 指定文件夹路径
origin_gt_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/GT/allGT'
relabeled_gt_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/GT/activeLearningRefinedAnnotateGT'
deep_learning_root_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/experimentRes/Contras'
# excel文件路径，excel记录了数据集划分，即某个视频是B数据集还是C数据集
excel_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/datasetAnalysis.xlsx'

# 要处理哪些实验结果
deep_learning_res_name_list = [ 'videoFeatTrainedWithDatasetA','video_feat_A_HS', 'video_feat_A_HS_shadow', 'video_feat_A_HS_shadowTraffic', 'video_feat_A_HS_allNewScene']
new_class_inExp = [[], [],[4],[4,5],[4,5,6]]


# 读取excel，判断某视频得到每个视频的是数据集B还是C
excel_file = pd.read_excel(excel_path,sheet_name='ActiveLearningDataset')
video_names_in_excel = excel_file.loc[:,'Video'].values # excel 中所有视频名
video_dataset_split = excel_file.loc[:,'Re-split'].values # 某视频是A还是B还是C

class_name_idx_map = {'Highway':0, 'Local':1, 'Ramp':2, 'Urban':3, 'shadow':4,'traffic':5,'tunnel':6}
new_class_names = ['shadow','traffic','tunnel']
old_class_name = ['Highway', 'Local', 'Ramp', 'Urban']

# 统计新类别在不同实验下的准确率
new_origin_component_dic = {}
for cls_name in new_class_names:
    new_origin_component_dic[cls_name] = {}
    for old_name in old_class_name:
        new_origin_component_dic[cls_name][old_name] = 0

# 逐个访问视频
for excel_row_idx, video_name in enumerate(video_names_in_excel):
    dataset_type = video_dataset_split[excel_row_idx]
    if dataset_type != 'C' and dataset_type != 'B': continue

    origin_gt = np.load(os.path.join(origin_gt_folder,video_name+'_gt.npy'))
    relabeld_gt = origin_gt.copy()

    if os.path.exists(os.path.join(relabeled_gt_folder,video_name+'_gt.npy')):
        relabeld_gt = np.load(os.path.join(relabeled_gt_folder,video_name+'_gt.npy'))
    
    for cls_name in new_class_names:
        cls_idx = class_name_idx_map[cls_name]
        cls_pos = relabeld_gt == cls_idx # 新类别cls_name所在的位置，true表示是新类别，否则不是

        if np.sum(cls_pos) == 0:continue

        gt = origin_gt.copy() # 当前实验对应的gt是什么
        tmp_gt = gt[cls_pos]
        for old_name in old_class_name:
            old_idx = class_name_idx_map[old_name]
            new_origin_component_dic[cls_name][old_name] += np.sum(tmp_gt == old_idx)


print('\n======================component======================')
print('ClassName', end = ' ')
for i in old_class_name:
    print(('%s'%i).ljust(8,' '), end = ' ')
print()

for new_name, old_item in new_origin_component_dic.items():
    print(('%s'%new_name).ljust(8,' '), end = ' ')
    for old_name, value in old_item.items():
        print('{0:>8}'.format(value), end = ' ')
    print()