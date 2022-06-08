from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity
from scipy import stats

cmap='jet'

xmin,xmax,ymin,ymax = -0.7790562102086777, 1.3152584934251397, -0.8115730743161789, 1.2091293048081884


# x = np.concatenate((np.linspace(xmin,xmax,100), np.linspace(xmin/6,xmax/2,100)))
# y = np.concatenate((np.linspace(ymin,ymax,100), np.linspace(ymin/2,ymax/3,100)))
# data = np.vstack([x,y]).T



classes = ['Highway','Local','Ramp','Urban']
c = 3
data = np.load('D:/Research/2021TrafficSceneClassification/res/Feat_reduced/Feat_reduced_'+classes[c] + '.npy')
n = np.shape(data)[0]
data = data[np.linspace(0,n-1,300,dtype=int),:]


dx = (xmax-xmin)/200
dy = (ymax-ymin)/200
X,Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
pos = np.vstack([X.ravel(), Y.ravel()]).T

kde = KernelDensity( bandwidth=0.1, kernel="gaussian")
kde.fit(data)
log_like = kde.score_samples(pos)
prob_dense = np.exp(log_like)
prob_dense *= dx*dy* n
Z = np.reshape(prob_dense, X.shape)



# kernel = stats.gaussian_kde(data.T)
# Z = np.reshape(kernel(pos.T).T, X.shape)



fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(np.rot90(Z), cmap=cmap,extent=[xmin,xmax,ymin,ymax], zorder = 0)
ax.scatter(data[:,0],data[:,1],s = 1)


fig = plt.figure(figsize=(6,6),dpi=600)
# fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap=cmap)
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.set_xticks([]) 
ax.set_yticks([]) 
ax.set_zticks([])




ax.set_zlim(0,5)

ax.view_init(30,-80)
print(ax.azim)
print(ax.elev)
plt.savefig('3D_probDense_%s.png'%classes[c], bbox_inches='tight', pad_inches=0)


# plt.show()



pass
