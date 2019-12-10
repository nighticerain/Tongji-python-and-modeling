import random as R
import math
x = 0
while 1:
    try:
        N = input('你想算几次')
        N = int(N)
        break
    except:
        print("输入的不是整数")

sinx_max = math.sin(x)

while N-1 >= 0:
    step = R.random()

    if R.randint(0,1) == 0:
        x1 = x - step
    else:
        x1 = x + step
    
    if 0< x1 <math.pi: 
        if math.sin(x1) > sinx_max:
            x = x1
            sinx_max = math.sin(x) 
        N = N - 1
    else:
        x1 = x
        continue

print('sin(x)在x={}时取得最大值{}'.format(x,sinx_max))

    





