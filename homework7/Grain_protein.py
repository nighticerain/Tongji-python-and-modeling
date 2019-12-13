import numpy as np
from PCR import PCR
data = np.loadtxt(r".\alldata.txt")
data = data.T
Y = data[:,-1]
X = data[:,:-1]
index = list(range(len(X)))
testInd = index[::10]

indexSet = set(index)
testSet = set(testInd)
trainSet = indexSet - testSet
trainInd = list(trainSet)
# print("测试样本编号：",testInd)
# print("训练样本编号：",trainInd)
X_test = X[testInd]
X_train = X[trainInd]
Y_test = Y[testInd]
Y_train = Y[trainInd]
pcr = PCR(X_train, Y_train)
pcr.confirmPCs()

# if there is no jump discontinuity, choose k by selecting t
# t reprensent how much information you loss

t = input("loss percent")
t = float(t)/100
k = 0
s_sum = sum(pcr.pca.lamda)
s = 0
for i in pcr.pca.lamda:
    s += i
    k += 1
    if s/s_sum >= 1-t:
        break

pcr.fit(k)
t = 1 - sum(pcr.pca.lamda[:k])/sum(pcr.pca.lamda)
print("loss percent:  ",t*100)

pcr.fTest(0.05)
y_hat = pcr.predict(X_test)
error = (y_hat - Y_test)/Y_test*100
print(error)



    