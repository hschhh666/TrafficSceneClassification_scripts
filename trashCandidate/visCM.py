import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

gt = np.load('C:/Users/A/Desktop/gt.npy')
pred = np.load('C:/Users/A/Desktop/lables.npy')

print(classification_report(gt,pred,target_names=['Highway','Local','Ramp','Urban']))

cm = confusion_matrix(gt, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Highway','Local','Ramp','Urban'])
disp.plot()
plt.show()