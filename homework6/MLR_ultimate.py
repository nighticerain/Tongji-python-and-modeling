import numpy as np
from scipy.stats import f 
class MLR:
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
    def fit(self):  # 求解回归系数 
        Xt= self.X.T
        XtX=np.dot (Xt,self.X)
        XtXinv = np.linalg.inv(XtX)
        temp = np.dot(XtXinv,Xt)
        self.A = np.dot(temp,self.Y)
    def getCoef(self):
        return self.A
    def predict(self,X):
        Y = np.dot(X,self.A)
        return Y

    def Ftest(self,alpha):  #MLR类的方法
        n=len(self.X)  # 样本数
        k=self.X.shape[-1]-1 # 变量个数
        f_arfa=f.isf(alpha, k, n-k-1)  # f临界值
        Yaver=self.Y.sum()/n
        Yhat=self.predict(self.X)  # 拟合的y值
        U=((Yhat-Yaver)**2).sum()
        Qe=((self.Y-Yhat)**2).sum()
        F=(U/k)/(Qe/(n-k-1))
        answer=[F,f_arfa,F>f_arfa]
        return answer
    
    def rCoef(self): #MLR类的方法
        n=len(self.X)
        Yaver=self.Y.sum()/n
        Yhat=self.predict(self.X)
        U=((Yhat-Yaver)**2).sum()
        Qe=((self.Y-Yhat)**2).sum()
        r=np.sqrt(U/(U+Qe))
        return r        

