
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from mpl_toolkits import mplot3d
from sklearn.neighbors.kde import KernelDensity


n = 1000
x1 = np.random.normal(scale = 0.2,size=n)
y1 = np.random.normal(scale=0.05, size=n)

n = 200
x2 = np.random.normal(loc=3,scale=0.02,size=n)
y2 = np.random.normal(loc=4,scale=0.03,size=n)

m1 = np.concatenate((x1,x2))
m2 = np.concatenate((y1,y2))


cmap = 'jet'
cmap = 'gist_earth_r'

colars = ['#ff0000','#70ad47','#9548a2','#02b0f0']
colars = ['#000000','#000000','#000000','#000000']
alphas = [0.1,0.3,0.5,0.2]

classes = ['Highway','Local','Ramp','Urban']
c = 0

data = np.load('D:/Research/2021TrafficSceneClassification/res/Feat_reduced/Feat_reduced_'+classes[c] + '.npy')
m1 = data[:,0]
m2 = data[:,1]

# ======================================

fig = plt.figure()
ax = fig.add_subplot(121)

ax.scatter(m1, m2, s=1, c=colars[c],alpha =  alphas[c], zorder = 1)
xmin = ax.viewLim.x0
xmax = ax.viewLim.x1
ymin = ax.viewLim.y0
ymax = ax.viewLim.y1
xmin,xmax,ymin,ymax = -0.7790562102086777, 1.3152584934251397, -0.8115730743161789, 1.2091293048081884

X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]

dx = (xmax-xmin)/200
dy = (ymax-ymin)/200

positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([m1, m2])
kernel = gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)


kde = KernelDensity( bandwidth=0.1, metric="euclidean", kernel="gaussian", algorithm="ball_tree")
kde.fit(data)
tmp = kde.score_samples(positions.T)
prob_dense = np.exp(tmp)
Z = np.reshape(prob_dense, X.shape)




ax.imshow(np.rot90(Z), cmap=cmap,extent=[xmin, xmax, ymin, ymax], zorder = 0)
plt.xlim((xmin,xmax))
plt.ylim((ymin,ymax))


ax2 = fig.add_subplot(122)
# ax2.scatter(m1, m2, s=1, c=colars[c],alpha = 1, zorder = 1)
clf = mixture.GaussianMixture(n_components=5, covariance_type="full")
clf.fit(values.T)

tmp = clf.score_samples(positions.T)
prob_dense = np.exp(tmp)

prob = prob_dense * dx * dy

print('prob sum = ', np.sum(prob_dense)*dx*dy)


Z = np.reshape(prob_dense, X.shape)
# Z = np.reshape(clf.score_samples(positions.T), X.shape)


ax2.scatter(m1, m2, s=1, c=colars[c],alpha = alphas[c], zorder = 1)


ax2.imshow(np.rot90(Z), cmap=cmap,extent=[xmin, xmax, ymin, ymax], zorder = 0)
plt.xlim((xmin,xmax))
plt.ylim((ymin,ymax))
plt.axis('off')  # 去掉坐标轴

# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap=cmap)
# ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
# ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
# ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
# ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
# ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
# ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
# ax.set_xticks([]) 
# ax.set_yticks([]) 
# ax.set_zticks([])





plt.show()