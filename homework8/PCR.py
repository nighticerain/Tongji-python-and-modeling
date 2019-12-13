import numpy as np
from PCA import PCA
from MLR_ultimate import MLR
class PCR:
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
    def confirmPCs(self):
        self.pca = PCA(self.X)
        compare = self.pca.SVDdecompose()
        return compare
    def fit(self,PCs):
        T,P = self.pca.PCAdecompose(PCs)
        print('T=',T)
        self.P = P
        oneCol = np.ones(T.shape[0])
        T = np.c_[oneCol,T] 
        self.mlr = MLR(T, self.Y)
        self.mlr.fit()
        self.A = self.mlr.getCoef()
    def predict(self,Xnew):
        T = np.dot(Xnew, self.P)
        oneCol=np.ones(T.shape[0])
        T = np.c_[oneCol,T]
        ans = self.mlr.predict(T)
        return ans
    def fTest(self,arfa):
        return self.mlr.Ftest(arfa)


