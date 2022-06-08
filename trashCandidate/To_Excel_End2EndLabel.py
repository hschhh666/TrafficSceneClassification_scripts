import numpy as np
import os
import openpyxl
from openpyxl import Workbook
from tqdm import tqdm

data_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/experimentRes/End2End/train'

f = []
for _,_,tmpf in os.walk(data_path):
    for i in tmpf:
        if i.endswith('_pred.npy'):
            f.append(i[:-9])

pbar = tqdm(f)
for name in pbar:
    pred_label = np.load(os.path.join(data_path, name+'_pred.npy'))
    wb = Workbook()
    ws = wb.active

    for i in range(np.shape(pred_label)[0]):
        l = pred_label[i]
        pos = 'A' + str(i+1)
        ws[pos] = l

    

    wb.save(os.path.join(data_path, name + '_End2EndPredLabel.xlsx'))
    wb.close()

print('Done')