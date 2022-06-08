import os
import numpy as np
import pandas as pd

deep_learning_root_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/experimentRes/Contras'
origin_gt_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/GT/allGT'
relabeled_gt_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/GT/activeLearningRefinedAnnotateGT4'
tar_gt_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/GT/activeLearningRefinedAnnotateGT5'

# excel文件路径，excel记录了数据集划分，即某个视频是B数据集还是C数据集
excel_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/datasetAnalysis.xlsx'

deep_learning_res_name_list = ['video_feat_A_HS_shadowTraffic2_P30', 'video_feat_A_HS_allNewScene']
exp_num = len(deep_learning_res_name_list)
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
    origin_gt = np.load(os.path.join(origin_gt_folder,video_name+'_gt.npy'))

    # if video_name == '201803291140_2018-03-29':
    #     print()

    if os.path.exists(os.path.join(relabeled_gt_folder,video_name+'_gt.npy')):
        relabeld_gt = np.load(os.path.join(relabeled_gt_folder,video_name+'_gt.npy'))
    else:
        relabeld_gt = origin_gt.copy()

    tar_gt = np.zeros_like(origin_gt)
    frame_num = np.shape(origin_gt)[0]

    preds = []
    
    for exp_idx, exp_name in enumerate(deep_learning_res_name_list):
        deep_learning_folder = os.path.join(deep_learning_root_folder, exp_name)
        pred = np.load(os.path.join(deep_learning_folder, video_name+'_predLabels.npy'))
        preds.append(pred)

    c = 5
    for i in range(frame_num):
        flag = False
        cnt = 0
        for j in range(exp_num):
            if preds[j][i] == c:
                cnt += 1
        if cnt >=2:
            flag = True
        if flag: # 预测值中存在标签为4的
            tar_gt[i] = c
        else:
            if relabeld_gt[i] == c:
                p = np.random.rand()
                if p < 0.0:
                    tar_gt[i] = c
                else:
                    tar_gt[i] = 7
                tar_gt[i] = origin_gt[i]
            else:
                tar_gt[i] = relabeld_gt[i]
    np.save(os.path.join(tar_gt_folder,video_name+'_gt.npy'), tar_gt)