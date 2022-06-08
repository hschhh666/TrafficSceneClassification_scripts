
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from mpl_toolkits import mplot3d


n = 400
x1 = np.random.normal(loc=-2,scale = 0.5,size=n)
y1 = np.random.normal(loc=-2,scale=1, size=n)

n = 400
x2 = np.random.normal(loc=2,scale=1,size=n)
y2 = np.random.normal(loc=2,scale=1,size=n)

m1 = np.concatenate((x1,x2))
m2 = np.concatenate((y1,y2))

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(m1, m2, s=1, c='k',alpha = 1, zorder = 1)
xmin = -5
xmax = 5
ymin = -5
ymax = 5

X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([m1, m2])




fig = plt.figure()
ax = plt.axes(projection='3d')
# ax2.scatter(m1, m2, s=1, c='k',alpha = 1, zorder = 1)
clf = mixture.GaussianMixture(n_components=2, covariance_type="full")
clf.fit(values.T)

tmp = clf.score_samples(positions.T)
prob = np.exp(tmp)

print(np.sum(prob))

Z = np.reshape(prob, X.shape)
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