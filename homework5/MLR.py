import numpy as np
from scipy.stats import f  
class MLR:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def fit(self):
        ones = np.ones(len(self.X))
        X = np.c_[ones, self.X]
        X_T = X.T
        a = (np.linalg.inv(np.dot(X_T, X))).dot(X_T)
        a = a.dot(self.Y)
        self.a = a
    def predict(self, X_test):
        ones = np.ones(len(X_test))
        X_test = np.c_[ones, X_test]
        Y_predict = (X_test).dot(self.a)
        return Y_predict
    def Ftest(self,alpha):
        y_hat = self.predict(self.X)
        Qe = np.sum((self.Y - y_hat)**2)
        ymean = self.Y.mean()
        U = np.sum((y_hat - ymean)**2)
        n = len(self.X)
        F = U / (Qe / (n-2))
        F_alpha = f.isf(alpha, 1, n-2)
        return [F,F_alpha,F>F_alpha]

alpha = 0.05
data = np.loadtxt(r'.\data.txt') 
# Y = 0.5 + 2.2 * X1 + 4.8 * X2 + (RAND() - 0.5) / 10
X = data[:,0:-1]
Y = data[:,-1]
mlr = MLR(X,Y)
mlr.fit()
Y_predict = mlr.predict(X)
print(Y_predict)
F_test = mlr.Ftest(alpha)
print(F_test)

        
    