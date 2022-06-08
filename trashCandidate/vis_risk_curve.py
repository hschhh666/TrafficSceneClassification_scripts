import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from torch import dtype
from tqdm import tqdm


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# ==========================输入参数==========================
video_name = '20160211_083030_2018-03-22'

# ==========================文件路径，不用改==========================
data_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/val'
risk_name = video_name+'_to_classes_risk'

# ==========================加载数据==========================
all_risk = np.load(os.path.join(data_path, risk_name + '.npy'))
cap = cv2.VideoCapture(os.path.join(data_path,video_name+'.mp4'))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
cap.release()


start = '02:19:28' # 时分秒
end = '02:19:42' # 时分秒

f = lambda x: (int(x.split(':')[0]) * 3600 + int(x.split(':')[1]) * 60 + int(x.split(':')[2])) * fps
start = f(start)
end = f(end)

start = 0
end = 500


fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(4):
    ax.plot(all_risk[start:end,i])

ax.legend(['Highway','Local','Ramp','Urban'])
plt.show()