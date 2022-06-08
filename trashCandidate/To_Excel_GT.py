import numpy as np
import os
import cv2
from openpyxl import Workbook
from tqdm import tqdm

vid_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/FINALDATA/VIDS'
csv_folder = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/FINALDATA/CSV'
gt_folder ='E:/Research/2021TrafficSceneClassification/Datasets/HSD/experimentRes/GT/allGT'

videos = []
for _,_,f in os.walk(vid_folder):
    for i in f:
        if i.endswith('mp4'):
            videos.append(i[:-4])
pass


pbar = tqdm(videos)
lst = -1
for name in pbar:
    csv = os.path.join(csv_folder,name+'.csv')
    gt_npy = []
    with open(csv) as f:
        lines = f.readlines()
        f.close()
        lines = lines[1:]
    cap = os.path.join(vid_folder,name+'.mp4')
    cap = cv2.VideoCapture(cap)
    fno = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert(fno == len(lines))

    wb = Workbook()
    ws = wb.active

    for i,line in enumerate(lines):
        l = int(line.split(',')[-2])       
        res = -1     
        if l == 1:
            res = 1
        elif l == 2:
            res = 0
        elif l == 3:
            res = 2
        elif l == 4:
            res = 3
        elif l == 5:
            res = 4
        else:
            if lst!= name:
                print('Error label in %s frame %d, label=%d'%(name,i, l))
                lst = name

        
        pos ='A'+str(i+1)
        ws[pos] = res
        gt_npy.append(res)

    gt_npy = np.array(gt_npy,dtype=int)
    np.save(os.path.join(gt_folder, name + '_gt.npy'), gt_npy)
    wb.save(os.path.join(gt_folder, name + '_gt.xlsx'))
    wb.close()

