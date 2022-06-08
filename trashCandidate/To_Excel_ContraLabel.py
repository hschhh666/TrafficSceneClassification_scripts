import numpy as np
import os
import openpyxl
from openpyxl import Workbook
from tqdm import tqdm

data_path = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/experimentRes/Contras/train'

f = []
for _,_,tmpf in os.walk(data_path):
    for i in tmpf:
        end = '_to_classes_risk.npy'
        if i.endswith(end):
            f.append(i[:-len(end)])

pbar = tqdm(f)
for name in pbar:
    file_path = os.path.join(data_path, name+end)
    risk = np.load(file_path)
    wb = Workbook()
    ws = wb.active
    for i in range(np.shape(risk)[0]):
        feat = risk[i,:]
        pred = np.argmin(feat)
        pos = 'A' + str(i+1)
        ws[pos] = pred

    wb.save(os.path.join(data_path, name + '_ContrasPredLabel.xlsx'))
    wb.close()

print('Done')