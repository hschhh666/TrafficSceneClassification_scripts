import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# 指定文件夹路径
origin_gt_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/GT/allGT'
relabeled_gt_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/GT/activeLearningRefinedAnnotateGT2'
deep_learning_root_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/experimentRes/Contras'
# excel文件路径，excel记录了数据集划分，即某个视频是B数据集还是C数据集
excel_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/datasetAnalysis.xlsx'

# 要处理哪些实验结果
deep_learning_res_name_list = [ 'videoFeatTrainedWithDatasetA','video_feat_A_HS', 'video_feat_A_HS_shadow', 'video_feat_A_HS_shadowTraffic2_P30', 'video_feat_A_HS_allNewScene']
new_class_inExp = [[], [],[4],[4,5],[4,5,6]]


# 读取excel，判断某视频得到每个视频的是数据集B还是C
excel_file = pd.read_excel(excel_path,sheet_name='ActiveLearningDataset')
video_names_in_excel = excel_file.loc[:,'Video'].values # excel 中所有视频名
video_dataset_split = excel_file.loc[:,'Re-split'].values # 某视频是A还是B还是C

class_name_idx_map = {'Highway':0, 'Local':1, 'Ramp':2, 'Urban':3, 'shadow':4,'traffic':5,'tunnel':6}
class_names = ['Highway', 'Local', 'Ramp', 'Urban','shadow','traffic','tunnel']



    
for exp_idx, exp_name in enumerate(deep_learning_res_name_list):
    deep_learning_folder = os.path.join(deep_learning_root_folder, exp_name)

    all_video_gt = np.empty((0),dtype=int)
    all_video_pd = np.empty((0),dtype=int)
    # 逐个访问视频
    for excel_row_idx, video_name in enumerate(video_names_in_excel):
        dataset_type = video_dataset_split[excel_row_idx]
        if dataset_type != 'B': continue

        origin_gt = np.load(os.path.join(origin_gt_folder,video_name+'_gt.npy'))
        relabeld_gt = origin_gt.copy()

        if os.path.exists(os.path.join(relabeled_gt_folder,video_name+'_gt.npy')):
            relabeld_gt = np.load(os.path.join(relabeled_gt_folder,video_name+'_gt.npy'))
        
        selected_frame = np.logical_and(relabeld_gt != -1, origin_gt != -1)
        # print('%s exclude frame percentage: %.0f%%'%(video_name, 100 - 100 * (np.sum(selected_frame) / np.shape(relabeld_gt)[0])))
        origin_gt = origin_gt[selected_frame]
        relabeld_gt = relabeld_gt[selected_frame]

        pred = np.load(os.path.join(deep_learning_folder, video_name+'_predLabels.npy'))
        risk = np.load(os.path.join(deep_learning_folder, video_name+'_risks.npy'))
        min_risk = np.min(risk, axis=1)

        # 排除relabeld_gt为7的数据，即排除异常数据
        pred = pred[selected_frame]
        min_risk = min_risk[selected_frame]

        gt = origin_gt.copy()
        for j in new_class_inExp[exp_idx]:
            gt[relabeld_gt == j] = j

        all_video_gt = np.concatenate((all_video_gt, gt))
        all_video_pd = np.concatenate((all_video_pd, pred))
    
    print('%s'%exp_name)
    target_names = [class_names[i] for i in range(4)] + [class_names[i] for i in new_class_inExp[exp_idx]]
    print(classification_report(all_video_gt, all_video_pd, target_names=target_names, digits=2))