import numpy as np
from MLR_ultimate import MLR
data = np.loadtxt(r'.\data_Cobb-Douglas.txt')
X = data[:,2:]
Y = data[:,1]
#add u
u = (np.random.rand(len(Y))-0.5)*10
Y_u = Y + u
ones = np.ones(X.shape[0])
lnX = np.c_[ones, np.log(X)]
lnY = np.log(Y_u)
mlr = MLR(lnX, lnY)
mlr.fit()
Y_predict = mlr.predict(lnX)
Q = np.exp(Y_predict)  
print(Y)
print(Q)



