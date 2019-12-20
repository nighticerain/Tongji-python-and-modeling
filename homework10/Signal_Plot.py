import numpy as np
x = y = np.arange(-np.pi,np.pi,0.5)
z = np.outer(np.cos(y),np.sin(x))

X, Y = np.meshgrid(x, y)
Z = z


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
fig1 = plt.figure(1)
ax = Axes3D(fig1)
ax.plot_surface(X, Y, Z)

ax.set_xlabel("x=sin")
ax.set_ylabel("y=cos")
ax.set_zlabel("z")

fig2 = plt.figure(2)
fig2 = plt.contour(X,Y,Z,10,colors='k')
plt.clabel(fig2, inline=1, fontsize=10) 

fig3 = plt.figure(3)
ax2 = Axes3D(fig3)
ax2.contour(X,Y,Z,10,colors='k')

plt.show()

