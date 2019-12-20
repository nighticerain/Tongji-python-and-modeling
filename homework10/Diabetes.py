import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
digits = load_diabetes()
X = digits.data
y = digits.target

y_std = np.std(y)
y_mean = np.mean(y)
y = (y - y_mean)/y_std

from sklearn.model_selection import KFold

err_list = []
for k in range(1,10):
    pls = PLSRegression(n_components = k)
    kf = KFold(n_splits=20)
    err = 0
    for train, test in kf.split(X):
        X_train, X_test, Y_train, Y_test = X[train,:], X[test,:], y[train], y[test]
        pls.fit(X_train,Y_train)
        YPred = pls.predict(X_test)[:,0] 
        err += np.mean((Y_test - YPred)/Y_test*100)
    err /= 20
    err_list.append(err)

n = err_list.index(min(err_list))+1
print("最佳主成分数为{0}".format(n))
