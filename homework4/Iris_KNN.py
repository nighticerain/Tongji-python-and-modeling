import math
import random

class point:
    def __init__(self,coor,label):
        self.coor = coor
        self.label = label
    def distance(self,p2):
        d = 0
        for i in range(len(self.coor)):
            d += (self.coor[i] - p2.coor[i]) ** 2
        self.d = math.sqrt(d)

class knn:
    def __init__(self, k):
        self.k = k
        
    def init_points(self, data_set, labels_set, test_data, test_lables):
        points = []
        for x,y in zip(data_set, labels_set):
            p = point(x,y) 
            points.append(p)
        self.points = points
        
        test_points = []
        for x,y in zip(test_data, test_lables):
            test_point = point(x,y) 
            test_points.append(test_point)
        self.test_points = test_points

    def identify(self):
        predicts = []
        ss = []
        for test_point in self.test_points:
            for p in self.points:
                p.distance(test_point)   
            Points_sorted = sorted(self.points, key = lambda o:o.d) 
            s = 0
            for i in range(self.k):
                s += Points_sorted[i].label / Points_sorted[i].d
                if s > 0:
                    predict = 1
                else:
                    predict = -1
            predicts.append(predict)
            ss.append(s) 
        self.predicts = predicts
        self.ss = ss     

    def kprint(self):
        for i in range(len(self.predicts)):
            if self.predicts[i] == self.test_points[i].label:
                con = '预测正确'
            else:
                con = '预测错误'
            print("点{0}的预测分类为{1},{2},s值为{3:.2f}".format(self.test_points[i].coor, self.predicts[i], con, self.ss[i]))


data_path = r'.\山-变色鸢尾花瓣.txt'
labels_path = r'.\山-变色鸢尾花瓣分类.txt'

f1 = open(data_path,'r')
lines = f1.readlines()
data_set = []
for line in lines:
    line = line.strip()
    unit = line.split()
    data = list(map(float, unit))
    data_set.append(data)
f1.close()

f2 = open(labels_path,'r')
lines = f2.readlines()
labels_set = []
for line in lines:
    line = line.strip()
    labels = float(line)
    labels_set.append(labels)
f2.close()

# 原数据集去重
del_i = []
for i in range(len(data_set)):
    for j in range(i+1, len(data_set)):
        if ((data_set[i])[1] == (data_set[j])[1]) and ((data_set[i])[0] == (data_set[j])[0]):
            del_i.append(j)

del_i = list(set(del_i))
del_i.sort(reverse=True)
print("原数据集中重复数据索引为:",del_i)
for i in del_i:
    del data_set[i]
    del labels_set[i]       


# 在数据集中选取五个点进行测试
n = 5
test_data = []
test_lables = []
for i in range(n):
    index = random.randint(0, len(data_set)-1)
    test_data.append(data_set[index])
    test_lables.append(labels_set[index])
    del data_set[index]
    del labels_set[index]


# 输入k值
while 1:
    try:
        k = input('请输入k = ')
        k = int(k)
        if k > len(data_set) / 10:
            print('k值过大，请重新输入')
        else:
            break
    except:
        print('输入异常')

knn1 = knn(k)
knn1.init_points(data_set, labels_set, test_data, test_lables)
knn1.identify()
knn1.kprint()




