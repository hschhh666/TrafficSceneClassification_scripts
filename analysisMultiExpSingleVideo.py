import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# 要处理的视频名
video_name = '201803291140_2018-03-29'


# 指定文件夹路径
origin_gt_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/GT/allGT'
relabeled_gt_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/GT/activeLearningRefinedAnnotateGT'
deep_learning_root_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/experimentRes/Contras'

# 要处理哪些实验结果
deep_learning_res_name_list = [ 'video_feat_A_HS_shadowTraffic', 'video_feat_A_HS_allNewScene']
new_class_inExp = [[4,5],[4,5,6]]

# 类别名与索引的对应关系
class_name_idx_map = {'Highway':0, 'Local':1, 'Ramp':2, 'Urban':3, 'shadow':4,'traffic':5,'tunnel':6}

all_time_window_risk_mean = []

for exp_idx, exp_name in enumerate(deep_learning_res_name_list):
    deep_learning_folder = os.path.join(deep_learning_root_folder, exp_name)
    origin_gt = np.load(os.path.join(origin_gt_folder,video_name+'_gt.npy'))
    origin_gt[origin_gt == -1] = 0
    
    if os.path.exists(os.path.join(relabeled_gt_folder,video_name+'_gt.npy')):
        relabeld_gt = np.load(os.path.join(relabeled_gt_folder,video_name+'_gt.npy'))
    else:
        relabeld_gt = origin_gt
    
    # gt = origin_gt
    # for j in new_class_inExp[exp_idx]:
    #     gt[relabeld_gt == j] = j
    gt = relabeld_gt.copy() # 当前实验对应的gt是什么
    for j in [4,5,6]:
        if j not in new_class_inExp[exp_idx]:
            gt[relabeld_gt == j] = origin_gt[relabeld_gt == j]


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
    print(np.mean(min_risk))
    # 计算一段时间窗内最小risk的均值方差
    time_window_size = 150 # 单位：帧
    time_window_risk_mean = np.zeros_like(min_risk)
    all_time_window_risk_mean.append(time_window_risk_mean)

    # 计算一定时间范围内的平均risk
    for i in range(frame_num):
        if i-time_window_size < 0 or i+time_window_size > frame_num: continue
        tmp_risk = min_risk[i-time_window_size:i+time_window_size]
        time_window_risk_mean[i] = np.mean(tmp_risk)


    # 画预测标签、真值标签和risk
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.scatter(list(range(frame_num)), pred, s=1)
    ax.scatter(list(range(frame_num)), gt+0.2, s=1)
    ax.set_title('pred vs gt')
    ax.legend(['Pred','GT'])
    ax = fig.add_subplot(212)
    ax.scatter(list(range(frame_num)), min_risk, s = 10)
    ax.plot(time_window_risk_mean,c='orange')
    ax.set_title('mean')
    ax.set_ylim(0,0.8)
    fig.suptitle(exp_name)


    # # 根据真值统计各类别risk的分布
    # class_idx = sorted(list(set(gt)))
    # class_names = [list(class_name_idx_map.keys())[c] for c in class_idx]
    # fig = plt.figure()
    # fig.suptitle('%s risk hist'%exp_name)
    # for j,c in enumerate(class_idx):
    #     ax = fig.add_subplot(1,len(class_idx), j+1)
    #     tmp_risk = risk[gt==c,:]
    #     tmp_risk = tmp_risk[:,c]
    #     ax.hist(tmp_risk)
    #     plt.title(class_names[j])
    


    # # 分类结果定量报告
    # cm = confusion_matrix(gt, pred)
    # tmp = list(set(list(gt) + list(pred)))
    # target_names = [list(class_name_idx_map.keys())[i] for i in tmp]
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = target_names)
    # disp.plot()
    # plt.title(exp_name)

    # print(deep_learning_res_name_list[exp_idx])
    # print(classification_report(gt, pred, target_names = target_names))




# 绘制不同实验下risk均值
fig = plt.figure()
for draw in all_time_window_risk_mean:
    plt.plot(draw)
plt.legend(deep_learning_res_name_list)

plt.show()


exit()

