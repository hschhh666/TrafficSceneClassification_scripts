# 根据risk，从数据集B中选择hard scene，即不添加新类别
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import os

# 各个类别的采样间隔
class_sample_interval = {'Highway':0.3,'Local':0.05,'Ramp':0.04, 'Urban':0.2} # 单位，秒

# 包含新类别的真值npy文件夹路径
activeLearningAnnotateGT_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/GT/activeLearningAnnotateGT'
# 生成的数据集文件夹路径
tarDataset_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/myDataset/activeLearningDataset_reprocess/hardScene'
# 视频文件夹路径
video_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/FINALDATA/VIDS'
# excel文件路径，excel记录了数据集划分，即某个视频是B数据集还是C数据集
excel_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/datasetAnalysis.xlsx'
# risk文件夹路径
deep_learning_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/experimentRes/Contras/videoFeatTrainedWithDatasetA'

class_name_idx_map = {'Highway':0,'Local':1,'Ramp':2, 'Urban':3}
for key in class_name_idx_map.keys():
    class_folder = os.path.join(tarDataset_folder, key)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

# 获取所有被标注过的视频文件名
video_names = []
for _,_, video_names in os.walk(activeLearningAnnotateGT_folder):
    break
video_names = [v[:-7] for v in video_names]

# 读取excel，判断某视频得到每个视频的是数据集B还是C
excel_file = pd.read_excel(excel_path,sheet_name='ActiveLearningDataset')
video_names_in_excel = excel_file.loc[:,'Video'].values # excel 中所有视频名
video_dataset_split = excel_file.loc[:,'Re-split'].values # 某视频是A还是B还是C

risk_threshold = 0.3

print('video', end = ' ')
for i in class_name_idx_map.keys():
    print(i, end=' ')
print('fps')

# 遍历视频，开始生成B数据集
for video_name in video_names:
    idx = np.where(video_names_in_excel == video_name)[0][0]
    dataset_type = video_dataset_split[idx]
    if dataset_type != 'B': continue # 确定当前视频是B数据集
    risk = os.path.join(deep_learning_folder, video_name+'_risks.npy')
    risk = np.load(risk)
    risk = np.min(risk, axis=1) # 获取risk

    cap = cv2.VideoCapture(os.path.join(video_folder,video_name+'.mp4'))
    fno = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    gt_npy = np.load(os.path.join(activeLearningAnnotateGT_folder, video_name + '_gt.npy'))
    print(video_name, end = ' ')
    for class_name, class_idx in class_name_idx_map.items():
        if class_idx not in gt_npy: 
            print(0, end = ' ')
            continue

        class_folder = os.path.join(tarDataset_folder, class_name)

        class_pos1 = set(np.where(gt_npy == class_idx)[0]) # 类别class_name在视频video_name中的帧号
        class_pos2 = set(np.where(risk > risk_threshold)[0])
        class_pos = class_pos1.intersection(class_pos2)
        class_pos = np.array(list(class_pos), dtype=int)

        print(len(class_pos), end = ' ')

        cur_interval_frame = int(class_sample_interval[class_name] * fps)
        sample_idx = np.array(list(range(0,np.shape(class_pos)[0], cur_interval_frame)), dtype=int)
        sample_class_idx = class_pos[sample_idx]
        sample_class_idx = list(sample_class_idx)

        pbar = tqdm(sample_class_idx,desc=class_name)
        for i in pbar:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret,frame = cap.read()
            if not ret:
                print('Read frame error in %s frame %d'%(video_name,i))
                continue
            frame = cv2.resize(frame,(400,224))
            img_path = video_name + '-' + str(i).rjust(10,'0')+'.jpg'
            img_path = os.path.join(class_folder, img_path)
            cv2.imwrite(img_path, frame)

    cap.release()