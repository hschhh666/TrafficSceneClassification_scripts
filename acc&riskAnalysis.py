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
deep_learning_res_name_list = [ 'videoFeatTrainedWithDatasetA','video_feat_A_HS2', 'video_feat_A_HS_shadow', 'video_feat_A_HS_shadowTraffic2_P30', 'video_feat_A_HS_allNewScene']
new_class_inExp = [[], [],[4],[4,5],[4,5,6]]


# 读取excel，判断某视频得到每个视频的是数据集B还是C
excel_file = pd.read_excel(excel_path,sheet_name='ActiveLearningDataset')
video_names_in_excel = excel_file.loc[:,'Video'].values # excel 中所有视频名
video_dataset_split = excel_file.loc[:,'Re-split'].values # 某视频是A还是B还是C

class_name_idx_map = {'Highway':0, 'Local':1, 'Ramp':2, 'Urban':3, 'shadow':4,'traffic':5,'tunnel':6}
new_class_names = ['Highway', 'Local', 'Ramp', 'Urban','shadow','traffic','tunnel']

# 统计新类别在不同实验下的准确率
newClass_exp_acc = {}
newClass_exp_risk = {}
for cls_name in new_class_names:
    newClass_exp_acc[cls_name] = {}
    newClass_exp_risk[cls_name] = {}
    for exp_name in deep_learning_res_name_list:
        newClass_exp_acc[cls_name][exp_name] = []
        newClass_exp_risk[cls_name][exp_name] = []

# 逐个访问视频
for excel_row_idx, video_name in enumerate(video_names_in_excel):
    dataset_type = video_dataset_split[excel_row_idx]
    if dataset_type != 'C': continue

    origin_gt = np.load(os.path.join(origin_gt_folder,video_name+'_gt.npy'))
    relabeld_gt = origin_gt.copy()

    if os.path.exists(os.path.join(relabeled_gt_folder,video_name+'_gt.npy')):
        relabeld_gt = np.load(os.path.join(relabeled_gt_folder,video_name+'_gt.npy'))
    
    for cls_name in new_class_names:
        cls_idx = class_name_idx_map[cls_name]
        cls_pos = relabeld_gt == cls_idx # 新类别cls_name所在的位置，true表示是新类别，否则不是

        if np.sum(cls_pos) == 0:continue

        for exp_idx, exp_name in enumerate(deep_learning_res_name_list):
            deep_learning_folder = os.path.join(deep_learning_root_folder, exp_name)
            pred = np.load(os.path.join(deep_learning_folder, video_name+'_predLabels.npy'))
            risk = np.load(os.path.join(deep_learning_folder, video_name+'_risks.npy'))
            min_risk = np.min(risk, axis=1)

            gt = relabeld_gt.copy() # 当前实验对应的gt是什么
            for j in [4,5,6]:
                if j not in new_class_inExp[exp_idx]:
                    gt[relabeld_gt == j] = origin_gt[relabeld_gt == j]
            
            # 现在，知道当前实验的gt了，也知道当前类的所有位置了
            tmp_gt = gt[cls_pos]
            tmp_pd = pred[cls_pos]
            tmp_risk = list(min_risk[cls_pos])

            cmp = tmp_gt == tmp_pd
            correct_cnt = np.sum(cmp)
            total_num = np.sum(cls_pos)
            
            newClass_exp_acc[cls_name][exp_name].append([correct_cnt, total_num])
            newClass_exp_risk[cls_name][exp_name] += tmp_risk


print('======================Acc======================')
print('ClassName', end = ' ')
for i in deep_learning_res_name_list:
    print(i, end= ' ')
print()

for c, exp in newClass_exp_acc.items():
    print(('%s'%c).ljust(8,' '), end = ' ')
    for exp_name, accList in exp.items():
        accList = np.array(accList)
        acc = 0
        if np.shape(accList)[0]:
            acc = int(100*np.sum(accList[:,0])/np.sum(accList[:,1]))
        print('{0:>3}%'.format(acc), end = ' ')
    print()


print('\n======================risk======================')
print('ClassName', end = ' ')
for i in deep_learning_res_name_list:
    print(i, end= ' ')
print()

for c, exp in newClass_exp_risk.items():
    print(('%s'%c).ljust(8,' '), end = ' ')
    for exp_name, riskList in exp.items():
        avg_risk = '%.3f'%np.average(riskList)
        print('{0:>5}'.format(avg_risk), end = ' ')
    print()


print('\nClassNumber:')
for c, exp in newClass_exp_acc.items():
    for exp_name, accList in exp.items():
        accList = np.array(accList)
        if np.shape(accList)[0]:
            total_num = np.sum(accList[:,1])
            print(total_num)
        break
        
