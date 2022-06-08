import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from torch import dtype
from tqdm import tqdm
import matplotlib.pyplot as plt

video_name = '201803231329_2018-03-23'

# ==========================文件路径，不用改==========================
data_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/val_center_no_norm'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

risk_name = video_name+'_to_classes_risk'
all_risk = np.load(os.path.join(data_path, risk_name + '.npy'))
cap = cv2.VideoCapture(os.path.join(data_path,video_name+'.mp4'))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))



fig = plt.figure()
ax = fig.add_subplot(111)

min_risk = np.min(all_risk, axis=1)
ax.plot(min_risk)



plt.show()
cap.release()
pass