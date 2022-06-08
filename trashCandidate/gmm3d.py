
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from mpl_toolkits import mplot3d
import os


classes = ['Highway','Local','Ramp','Urban']
c = 3

data = np.load('D:/Research/2021TrafficSceneClassification/res/Feat_reduced/Feat_reduced_'+classes[c] + '.npy')
m1 = data[:,0]
m2 = data[:,1]



# =================================

fig = plt.figure()
ax = fig.add_subplot(121)

ax.scatter(m1, m2, s=1, c='k',alpha = 1, zorder = 1)
xmin = ax.viewLim.x0
xmax = ax.viewLim.x1
ymin = ax.viewLim.y0
ymax = ax.viewLim.y1

X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([m1, m2])

dx = (xmax-xmin)/200
dy = (ymax-ymin)/200


# fig = plt.figure()
# ax = plt.axes(projection='3d')
ax = fig.add_subplot(122,projection='3d')
# ax2.scatter(m1, m2, s=1, c='k',alpha = 1, zorder = 1)
clf = mixture.GaussianMixture(n_components=5, covariance_type="full")
clf.fit(values.T)

tmp = clf.score_samples(positions.T)
prob_dense = np.exp(tmp)

print('prob sum = ', np.sum(prob_dense)*dx*dy)

Z = np.reshape(prob_dense, X.shape)
Z = np.reshape(clf.score_samples(positions.T), X.shape)


# ax2.imshow(np.rot90(Z), cmap=plt.cm.gist_earth,extent=[xmin, xmax, ymin, ymax], zorder = 0)
# plt.xlim((xmin,xmax))
# plt.ylim((ymin,ymax))
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='jet')



ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 

# Get rid of the spines
ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

# ax.xaxis._axinfo["grid"]['color'] = "#000000"
# ax.xaxis._axinfo["grid"]['linestyle'] = ":"
# ax.yaxis._axinfo["grid"]['color'] = "#000000"
# ax.yaxis._axinfo["grid"]['linestyle'] = ":"
# ax.zaxis._axinfo["grid"]['color'] = "#000000"
# ax.zaxis._axinfo["grid"]['linestyle'] = ":"

# Get rid of the ticks
ax.set_xticks([]) 
ax.set_yticks([]) 
ax.set_zticks([])

# ax.set_title('surface')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

plt.show()