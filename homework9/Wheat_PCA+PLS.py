import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt(r".\wheat_train_PCA_X.txt")
B = np.loadtxt(r".\wheat_train_PCA_Y.txt")


from PCA import PCA
aver = A.mean(axis=0)
std = A.std(axis=0)
X = (A-aver) /std
pca = PCA(X)
print(pca.SVDdecompose())
T,P = pca.PCAdecompose(6)

cls1 = B == 1.0
cls2 = B != 1.0

plt.subplot(1,2,1)
plt.plot(T[cls1,0],T[cls1,1],'ro')
plt.plot(T[cls2,0],T[cls2,1],'b^')


#PLSR
from sklearn.cross_decomposition import PLSRegression
pls = PLSRegression(n_components = 3, scale=True)
pls.fit(A, B)

T = pls.x_scores_

plt.subplot(1,2,2)
plt.plot(T[cls1,0],T[cls1,1],'ro')
plt.plot(T[cls2,0],T[cls2,1],'b^')

plt.show()