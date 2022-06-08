import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# 指定文件夹路径
origin_gt_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/GT/allGT'
relabeled_gt_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/GT/activeLearningAnnotateGT'
deep_learning_root_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/experimentRes/Contras'
# excel文件路径，excel记录了数据集划分，即某个视频是B数据集还是C数据集
excel_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/datasetAnalysis.xlsx'

# 要处理哪些实验结果
deep_learning_res_name_list = [ 'videoFeatTrainedWithDatasetA','video_feat_A_HS','video_feat_A_HS_allNewScene']

# 读取excel，判断某视频得到每个视频的是数据集B还是C
excel_file = pd.read_excel(excel_path,sheet_name='ActiveLearningDataset')
video_names_in_excel = excel_file.loc[:,'Video'].values # excel 中所有视频名
video_dataset_split = excel_file.loc[:,'Re-split'].values # 某视频是A还是B还是C

isFind = False
# isFind = True

if isFind:
    # 逐个访问视频
    for excel_row_idx, video_name in enumerate(video_names_in_excel):
        dataset_type = video_dataset_split[excel_row_idx]
        if dataset_type != 'C': continue

        all_time_window_risk_mean = []

        # 计算每个实验的情况
        for exp_idx, exp_name in enumerate(deep_learning_res_name_list):
            deep_learning_folder = os.path.join(deep_learning_root_folder, exp_name)

            # 读取文件
            risk = os.path.join(deep_learning_folder, video_name+'_risks.npy')
            pred = os.path.join(deep_learning_folder, video_name+'_predLabels.npy')
            risk = np.load(risk)
            pred = np.load(pred)

            # 视频总帧数
            frame_num = np.shape(risk)[0]

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
            
            all_time_window_risk_mean.append(time_window_risk_mean)

        # 绘制不同实验下risk均值
        print(video_name)
        fig = plt.figure()
        for draw in all_time_window_risk_mean:
            plt.plot(draw)
        plt.legend(deep_learning_res_name_list)
        plt.show()
        plt.close()


else:
    video_name = '20180423_103555_2018-04-23'
    start_idx = 21400
    end_idx = 22150

    # 计算每个实验的情况
    all_time_window_risk_mean = []
    all_min_risk = []
    for exp_idx, exp_name in enumerate(deep_learning_res_name_list):
        deep_learning_folder = os.path.join(deep_learning_root_folder, exp_name)

        # 读取文件
        risk = os.path.join(deep_learning_folder, video_name+'_risks.npy')
        pred = os.path.join(deep_learning_folder, video_name+'_predLabels.npy')
        risk = np.load(risk)
        pred = np.load(pred)

        if exp_name in ['videoFeatTrainedWithDatasetA','video_feat_A_HS']:
            gt = np.load(os.path.join(origin_gt_folder,video_name+'_gt.npy'))
        else:
            # 获取真值，如果有新标注的则读取新标注真值，否则读取原真值
            if os.path.exists(os.path.join(relabeled_gt_folder,video_name+'_gt.npy')):
                gt = np.load(os.path.join(relabeled_gt_folder,video_name+'_gt.npy'))
            else:
                gt = np.load(os.path.join(origin_gt_folder,video_name+'_gt.npy'))
        
    
        # 视频总帧数
        frame_num = np.shape(risk)[0]

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
        
        all_time_window_risk_mean.append(time_window_risk_mean[start_idx:end_idx])
        
        all_min_risk.append(min_risk[start_idx:end_idx])
        clip_acc = 100*accuracy_score(gt[start_idx:end_idx], pred[start_idx:end_idx])
        print('Acc %.0f%%'%clip_acc)


fig = plt.figure(figsize=(18, 6), dpi=100)
colors = ['r','g','b','orange']
for i, exp_name in enumerate(deep_learning_res_name_list):
    ax = fig.add_subplot(10 + len(deep_learning_res_name_list)*100 + i+1)
    ax.scatter(range(np.shape(all_min_risk[i])[0]), all_min_risk[i], s = 6, c = colors[i])
    # plt.plot(all_time_window_risk_mean[i], c='orange')

    ax.set_ylim(-0.1,0.6)
    plt.xticks([])  #去掉横坐标值
    plt.yticks([])  #去掉纵坐标值
    plt.axis('off')  # 去掉坐标轴



plt.show()