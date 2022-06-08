import os
import numpy as np
import pandas as pd

relabeled_gt_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/GT/activeLearningRefinedAnnotateGT'
tar_gt_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/GT/activeLearningRefinedAnnotateGT2'

# excel文件路径，excel记录了数据集划分，即某个视频是B数据集还是C数据集
excel_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/datasetAnalysis.xlsx'

if not os.path.exists(tar_gt_folder):
    os.makedirs(tar_gt_folder)


# 读取excel，判断某视频得到每个视频的是数据集B还是C
excel_file = pd.read_excel(excel_path,sheet_name='ActiveLearningDataset')
video_names_in_excel = excel_file.loc[:,'Video'].values # excel 中所有视频名
video_dataset_split = excel_file.loc[:,'Re-split'].values # 某视频是A还是B还是C
# 逐个访问视频
for excel_row_idx, video_name in enumerate(video_names_in_excel):
    dataset_type = video_dataset_split[excel_row_idx]
    if dataset_type != 'B' and dataset_type != 'C': continue

    if os.path.exists(os.path.join(relabeled_gt_folder,video_name+'_gt.npy')):
        relabeld_gt = np.load(os.path.join(relabeled_gt_folder,video_name+'_gt.npy'))
    else:continue
    relabeld_gt[relabeld_gt == 7] = -1
    np.save(os.path.join(tar_gt_folder,video_name+'_gt.npy'), relabeld_gt)