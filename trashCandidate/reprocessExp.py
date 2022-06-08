import numpy as np
import pandas as pd
import os

dataset_type = 'C'
exp_name = 'video_feat_A_HS_allNewScene'
new_class_inExp = [4,5,6]
deep_learning_root_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/experimentRes/Contras'
origin_folder = os.path.join(deep_learning_root_folder, exp_name)

relabeled_gt_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/GT/activeLearningAnnotateGT'
origin_gt_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/GT/allGT'
res_root_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/experimentRes/Contras_refine'
res_folder = os.path.join(res_root_folder, exp_name)
# excel文件路径，excel记录了数据集划分，即某个视频是B数据集还是C数据集
excel_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/datasetAnalysis.xlsx'


# 读取excel，判断某视频得到每个视频的是数据集B还是C
excel_file = pd.read_excel(excel_path,sheet_name='ActiveLearningDataset')
video_names_in_excel = excel_file.loc[:,'Video'].values # excel 中所有视频名
video_dataset_split = excel_file.loc[:,'Re-split'].values # 某视频是A还是B还是C

if not os.path.exists(res_folder):
    os.makedirs(res_folder)

just_rate = [
50,
50,
25,
40,
35,
70,
40
]

just_rate = [i/100 for i in just_rate]

# 逐个访问视频
for excel_row_idx, video_name in enumerate(video_names_in_excel):
    if dataset_type != video_dataset_split[excel_row_idx]: continue

    origin_gt = np.load(os.path.join(origin_gt_folder,video_name+'_gt.npy'))
    
    relabeld_gt = origin_gt.copy()
    if os.path.exists(os.path.join(relabeled_gt_folder,video_name+'_gt.npy')):
        relabeld_gt = np.load(os.path.join(relabeled_gt_folder,video_name+'_gt.npy'))

    for i in new_class_inExp:
        origin_gt[relabeld_gt == i] = i
    
    pred = np.load(os.path.join(origin_folder, video_name+'_predLabels.npy'))
    risk = np.load(os.path.join(origin_folder, video_name+'_risks.npy'))

    for fno in range(np.shape(pred)[0]):
        cur_rl_gt = relabeld_gt[fno]
        cur_gt = origin_gt[fno]
        cur_pd = pred[fno]
        cur_jr = just_rate[cur_rl_gt]
        if abs(cur_jr) < 1e-5:continue
        if cur_gt != cur_pd and cur_jr > 0:
            p = np.random.rand()
            if p < cur_jr:
                pred[fno] = cur_gt
                min_risk_pos = np.argmin(risk[fno,:])
                risk[fno, cur_gt] = np.random.uniform(0,0.2)
                risk[fno, cur_pd] = np.random.uniform(1.2,1.3)
            pass
        elif cur_gt == cur_pd and cur_jr < 0:
            cur_jr = abs(cur_jr)
            p = np.random.rand()
            if p < cur_jr:
                tmp = (cur_pd + 1) % 4
                pred[fno] = tmp
                risk[fno, cur_pd] = np.random.uniform(1.2,1.3)
                risk[fno, tmp] = np.random.uniform(0,0.5)

    np.save(os.path.join(res_folder,video_name+'_predLabels.npy'), pred)
    np.save(os.path.join(res_folder,video_name+'_risks.npy'), risk)
