import math

# define class
class point:
    def __init__(self,coor,label):
        self.coor = coor
        self.label = label
    def distance(self,p2):
        d = 0
        for i in range(len(self.coor)):
            d += (self.coor[i] - p2.coor[i]) ** 2
        self.d = math.sqrt(d)

# data
X1 = [[5.1, 3.5 ],  [4.9, 3. ],  [4.7, 3.2], [4.6, 3.1], [5. , 3.6], [5.4, 3.9],
       [4.6, 3.4], [5. , 3.4 ], [4.4, 2.9], [4.9, 3.1]]
X2 = [[5.5, 2.6 ],  [6.1, 3. ],  [5.8, 2.6], [5. , 2.3], [5.6, 2.7],
       [5.7, 3. ],  [5.7, 2.9],  [6.2, 2.9], [5.1, 2.5],  [5.7, 2.8]]
x1, x2 = X1.pop(), X2.pop()#注意有无括号有区别
X1.extend(X2)
X = X1

Y1 = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
Y2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
y1, y2 = Y1.pop(), Y2.pop()
Y1.extend(Y2)
Y = Y1

# process data
PX = []
for i in range(len(X)):
    Pi = point(X[i],Y[i]) 
    PX.append(Pi)
Px1, Px2 = point(x1,y1), point(x2,y2)

# Knn
def knn_id(point,PX,k):
    for x in PX:
        x.distance(point)   
    PXn = sorted(PX,key = lambda o:o.d)
    s = 0
    for i in range(k):
        s += PXn[i].label/ PXn[i].d
    if s > 0:
        label = 1
    else:
        label = -1
    return label,s

# input K
while 1:
    try:
        k = input('请输入K = ')
        k = int(k)
        if k > len(X):
            print('k值过大，请重新输入')
        else:
            break
    except:
        print('输入异常')

    
# print
def knn_print(Px,label):
    if label == Px.label:
        con = '预测正确'
    else:
        con = '预测错误'
    print("点{0}的预测分类为{1},{2}".format(Px.coor,label,con))
    
label1,s1 = knn_id(Px1, PX, k)
knn_print(Px1,label1)
print("s值为",s1)
label2,s2 = knn_id(Px2, PX, k)
knn_print(Px2,label2)
print("s值为",s2)


