import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from scipy.special import softmax

video_name = '201803281038_2018-03-28'

# contras_risk = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/experimentRes/Contras/train/%s_Contras_gmm_risk.npy'%video_name
gt = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/FINALDATA/CSV/%s.csv'%video_name

contras_risk = 'E:/Research/2021TrafficSceneClassification/Datasets/HSD/experimentRes/Contras/val/201803281038_2018-03-28_gmm_risk.npy'


contras_risk = np.load(contras_risk)

with open(gt) as f:
    lines = f.readlines()
    f.close()
    lines = lines[1:]

gt = []
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
    gt.append(res)

gt = np.array(gt,dtype=int)
contra_pred = []

for i in range(np.shape(contras_risk)[0]):
    r = contras_risk[i,:]
    p = np.argmin(r)
    contra_pred.append(p)
contra_pred = np.array(contra_pred, dtype=int)

end2end_pred = np.load('E:/Research/2021TrafficSceneClassification/Datasets/HSD/experimentRes/End2End/val/%s_pred.npy'%video_name)
end2end_feat = np.load('E:/Research/2021TrafficSceneClassification/Datasets/HSD/experimentRes/End2End/val/%s_feat.npy'%video_name)

x = list(range(np.shape(end2end_pred)[0]))



cname = ["Highway",'Local','Ramp','Urban']


fig = plt.figure()
for i in range(4):
    ax = fig.add_subplot(22*10+i+1)
    r = contras_risk[gt == i,i]
    ax.hist(r)
    ax.set_title(cname[i])
fig.suptitle('Contras '+video_name)

delta = 0.05
s = 1
plt.figure()
plt.scatter(x,contra_pred, zorder=0,s=s)
plt.scatter(x,end2end_pred-delta,zorder=1,s=s)
plt.scatter(x,gt+delta,zorder=2,s=s)
plt.legend(['Contras','e2e','gt'])
plt.title(video_name)


end2end_prob = softmax(end2end_feat,axis=1)
end2end_risk = 1 - end2end_prob
fig = plt.figure()
for i in range(4):
    ax = fig.add_subplot(22*10+i+1)
    r = end2end_risk[gt == i,i]
    ax.hist(r)
    ax.set_title(cname[i])
fig.suptitle('E2E '+video_name)



print('=================contras=================')
print(classification_report(gt, contra_pred,target_names=cname))

print('=================e2e=================')
print(classification_report(gt, end2end_pred,target_names=cname))
plt.show()
pass