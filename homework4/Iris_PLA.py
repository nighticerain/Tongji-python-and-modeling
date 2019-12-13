import matplotlib.pyplot as plt

def lossF(x,y,a,b):  # 计算损失值 x-sample y-label 
    temp=[]
    for one in x:
        temp1=[one[i]*a[i] for i in range(len(one))]
        temp1=sum(temp1)+b   # 每个样本的 a1x1+a2x2+b的取值
        temp.append(temp1)
    temp2=[]
    for i in range(len(temp)):
        temp2.append(temp[i]*y[i])   # 每个样本函数值与真值的乘积
    s=0
    for i in temp2:
        s += max(0,-i)   # 计算损失函数，每项取负值，即误判样本
    return s,temp2  # temp2返回有用，用于找误判，调节系数

# Iris data
data_path = r'.\山-变色鸢尾花瓣.txt'
labels_path = r'.\山-变色鸢尾花瓣分类.txt'

f1 = open(data_path,'r')
lines = f1.readlines()
x = []
for line in lines:
    line = line.strip()
    unit = line.split()
    data = list(map(float, unit))
    x.append(data)
f1.close()

f2 = open(labels_path,'r')
lines = f2.readlines()
y = []
for line in lines:
    line = line.strip()
    labels = float(line)
    y.append(labels)
f2.close()

# 原数据集去重
del_i = []
for i in range(len(x)):
    for j in range(i+1, len(x)):
        if ((x[i])[1] == (x[j])[1]) and ((x[i])[0] == (x[j])[0]):
            del_i.append(j)

del_i = list(set(del_i))
del_i.sort(reverse=True)

for i in del_i:
    del x[i]
    del y[i]       

#learn
a=[1,1]# 初始化a、b系数，设定学习速率为0.1
b=1
rate=0.1
for i in range(100):#根据梯度下降原理对a、b系数进行修正，直到损失值最小
    loss,temp=lossF(x,y,a,b)
    print("%.3f" %loss)
    for j in range(len(temp)):
        if temp[j] <0:#寻找误判样本
            a[0] +=rate*y[j]*x[j][0]#更新a1
            a[1] +=rate*y[j]*x[j][1]#更新a2
            b +=rate*y[j]#更新b
print("model is {:8.3f}x1+{:8.3f}x2+{:8.3f}".format(a[0],a[1],b))


#plot
xvalue=[x[i][0] for i in range(len(x))]
xmin=min(xvalue)
xmax=max(xvalue)
xp=[xmin,xmax]
yp=[-a[0]/a[1]*xmin-b/a[1],-a[0]/a[1]*xmax-b/a[1]]

cls1x=[x[i][0] for i in range(len(x)) if y[i]==-1]
cls2x=[x[i][0] for i in range(len(x)) if y[i]==1]
cls1y=[x[i][1] for i in range(len(x)) if y[i]==-1]
cls2y=[x[i][1] for i in range(len(x)) if y[i]==1]
plt.plot(cls1x,cls1y,'b^')
plt.plot(cls2x,cls2y,'r^')
plt.plot(xp,yp)
plt.show()

print("原数据集中重复数据索引为:",del_i)