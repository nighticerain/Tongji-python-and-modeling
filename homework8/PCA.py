import numpy as np
class PCA:
    def __init__(self, X):
        self.X = X
    def SVDdecompose(self):
        B = np.linalg.svd(self.X, full_matrices=False)
        self.lamda = lamda = B[1]
        self.P = B[2].T
        self.T = B[0]*B[1]
        compare = [lamda[i]/lamda[i+1] for i in range(len(lamda)-1)]
        return compare
    def PCAdecompose(self,k):  
        T = self.T[:,:k]
        P = self.P[:,:k]
        return T,P

