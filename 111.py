import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 载入数据
df = pd.read_csv('D:\wisteria\data.csv', header=None)
print(df)
# x转成数组
px = np.array(df.iloc[:, 0])
# y转成数组
py = np.array(df.iloc[:, 1])
plt.scatter(px,py)
# plt.show()

# --------------模型搭建-----------------
# 模型函数
def f(x,k,b):
    return k*x+b

# 损失函数
def L(x,y,k,b):
    s = 0
    n = len(x)
    for i in range(len(x)):
        s += (k*x[i]+b-y[i])**2
    return s/2

# k梯度
def grad_k(x,y,k,b):
    s = 0
    n = len(x)
    for i in range(n):
        s += (k*x[i]+b-y[i])*x[i]
    return 2/n * s

# b梯度
def grad_b(x,y,k,b):
    s = 0
    n = len(x)
    for i in range(n):
        s += (k*x[i]+b-y[i])
    return 2/n * s

# --------------梯度下降-----------------
# 初始化参数
k = 0
b = 0

times = 10		# 训练次数
alpha = 0.0004	# 步长

for i in range(times):
    # 梯度下降迭代
    k = k - alpha*grad_k(px,py,k,b)
    b = b - alpha*grad_b(px,py,k,b)
    if i% 10 == 0:
        print("%d:%.2f"%(i,L(px,py,k,b)))	# 输出当前损失函数

plt.plot(px,f(px,k,b),c = "red")	# 绘制训练结果
plt.show()

