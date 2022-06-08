import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# 要处理的视频名
video_name = '20160211_083140_2018-03-14'

# 指定文件夹路径
gt_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/GT/activeLearningAnnotateGT'
deep_learning_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/experimentRes/Contras/video_feat_A_HS_allNewScene'

# 读取文件
risk = os.path.join(deep_learning_folder, video_name+'_risks.npy')
pred = os.path.join(deep_learning_folder, video_name+'_predLabels.npy')
feat = os.path.join(deep_learning_folder, video_name+'_memoryFeature.npy')
gt   = os.path.join(gt_folder,video_name+'_gt.npy')
risk = np.load(risk)
pred = np.load(pred)
feat = np.load(feat)
feat = feat / np.linalg.norm(feat,axis=1, keepdims=True) # 撒币了，在计算特征的时候忘了归一化了
gt = np.load(gt)

# 视频总帧数
frame_num = np.shape(gt)[0]
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


# 根据真值统计各类别risk的分布
class_num = 4
class_names = ['Highway','Local','Ramp','Urban','shadow','traffic','tunnel']
fig = plt.figure()
fig.suptitle('risk hist')
for i in range(class_num):
    ax = fig.add_subplot(1,class_num, i+1)
    tmp_risk = risk[gt==i,:]
    tmp_risk = tmp_risk[:,i]
    ax.hist(tmp_risk)
    plt.title(class_names[i])


# 分类结果定量报告
cm = confusion_matrix(gt, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = class_names)
disp.plot()


plt.show()


exit()

# 根据预测值画risk分布
c = 1
tmp = risk[pred == c, c]
fig = plt.figure()
ax = fig.add_subplot(211)
ax.set_title('pred risk')
f = ax.hist(tmp)
ymax = max(f[0])
ymax *=1.05
ax.set_ylim(0,ymax)
ax.set_xlim(0,1)
ax = fig.add_subplot(212)
ax.scatter(list(range(np.shape(tmp)[0])), tmp)


# 对比异常类别前后的risk分布
origin_risk = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/experimentRes/Contras/videoFeatFinGrained/' + video_name+'_risks.npy'
origin_risk = np.load(origin_risk)
origin_min_risk = np.min(origin_risk, axis=1)

fig = plt.figure()
ax1 = fig.add_subplot(122)
ax1.set_title('pred risk')
tmp = risk[pred == c, c]
# tmp = risk[48592:48730,c]
f = ax1.hist(tmp,range=(0,1),bins=10)
ymax = max(f[0])

ax2 = fig.add_subplot(121)
tmp = origin_min_risk[pred == c]
# tmp=origin_min_risk[48592:48730]
f = ax2.hist(tmp,range=(0,1),bins=10)
ymax = max(max(f[0]), ymax)
ymax *= 1.05
ax2.set_title('origin risk')
ax1.set_ylim(0,ymax)
ax1.set_xlim(0,1)
ax2.set_ylim(0,ymax)
ax2.set_xlim(0,1)


plt.show()