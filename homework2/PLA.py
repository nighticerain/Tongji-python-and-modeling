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
x1=[[5.1, 3.5 ],  [4.9, 3. ],  [4.7, 3.2], [4.6, 3.1], [5. , 3.6], [5.4, 3.9],
       [4.6, 3.4], [5. , 3.4 ], [4.4, 2.9], [4.9, 3.1]]
x2=[[5.5, 2.6 ],  [6.1, 3. ],  [5.8, 2.6], [5. , 2.3], [5.6, 2.7],
       [5.7, 3. ],  [5.7, 2.9],  [6.2, 2.9], [5.1, 2.5],  [5.7, 2.8]]
x1.extend(x2)
x=x1
y=[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
y1=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
y.extend(y1)

#learn
a=[1,1]# 初始化a、b系数，设定学习速率为0.1
b=1
rate=0.1
for i in range(100):#根据梯度下降原理对a、b系数进行修正，直到损失值最小
    loss,temp=lossF(x,y,a,b)
    print(loss)
    for j in range(len(temp)):
        if temp[j] <0:#寻找误判样本
            a[0] +=rate*y[j]*x[j][0]#更新a1
            a[1] +=rate*y[j]*x[j][1]#更新a2
            b +=rate*y[j]#更新b
print("model is {:10.3f}x1+{:10.3f}x2+{:10.3f}".format(a[0],a[1],b))


#plot
xvalue=[x[i][0] for i in range(len(x))]
xmin=min(xvalue)
xmax=max(xvalue)
xp=[xmin,xmax]
yp=[-a[0]/a[1]*xmin-b/a[1],-a[0]/a[1]*xmax-b/a[1]]
from pylab import *
cls1x=[x[i][0] for i in range(len(x)) if y[i]==-1]
cls2x=[x[i][0] for i in range(len(x)) if y[i]==1]
cls1y=[x[i][1] for i in range(len(x)) if y[i]==-1]
cls2y=[x[i][1] for i in range(len(x)) if y[i]==1]
plot(cls1x,cls1y,'b^')
plot(cls2x,cls2y,'r^')
plot(xp,yp)
show()

