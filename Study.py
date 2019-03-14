import numpy as np
import matplotlib as plt
account = 'qiyue'
password = '123456'

print('pkease input account')
user_account = input('please input:')
print('please input password')
user_password = input()
print(type(password))
""" def main():
    pass

if __name__ == '__main__':
    main() """
# 直方图
import pylab
import random
SAMPLE_SIZE = 100
real_rand_vars = []
real_rand_vars = [random.random() for _ in range(SAMPLE_SIZE)]
pylab.hist(real_rand_vars, 10)
pylab.xlabel("Number range")

#coding: utf-8
import random
import matplotlib
import matplotlib.pyplot as plt

SAMPLE_SIZE = 1000
# histogram buckets
buckets = 100
plt.figure()
matplotlib.rcParams.update({'font.size': 7})
# 第一张图[0,1）之间分布的随机变量
plt.subplot(621) #6*2 第一张图
plt.xlabel("random.random")
# Return the next random floating point number is the range (0.0, 1.0)
res = [random.random() for _ in range(1, SAMPLE_SIZE)]
plt.hist(res, buckets)
# 第二张图 均匀分布的随机变量
plt.subplot(622)
plt.xlabel("random.uniform")
# Return a random floating point number N such that a <= N <= b for a <= b and b <= N  <= a for b < a
a = 1
b = SAMPLE_SIZE
res = [random.uniform(a,b) for _ in range(1,SAMPLE_SIZE)]
plt.hist(res,buckets)

#第三张图 三角形分布
plt.subplot(623)
plt.xlabel("random.triangular")
# Return a random floating point number N such that low <= N <= high and with the specified
low = 1
high = SAMPLE_SIZE
res = [random.triangular(low,high) for _ in range(1, SAMPLE_SIZE)]
plt.hist(res,buckets)

# 第四张图 beta分布,参数的条件是alpha和beta都大于0，返回值再0~1之间。
plt.subplot(624)
plt.xlabel("random.betavariate")
alpha = 1
beta = 10
res = [random.betavariate(alpha,beta) for _ in range(1, SAMPLE_SIZE)]
plt.hist(res, buckets)

# 第五张图 指数分布
plt.subplot(625)
plt.xlabel("random.expovariate")
lambd = 1.0 /((SAMPLE_SIZE + 1) /2.)
res = [random.expovariate(lambd) for _ in range(1, SAMPLE_SIZE)]
plt.hist(res, buckets)

#第六张图 gamma分布
plt.subplot(626)
plt.xlabel("random.gammavariate")
alpha = 1
beta = 10
res = [random.gammavariate(alpha,beta) for _ in range(1, SAMPLE_SIZE)]
plt.hist(res, buckets)

#第七张图 对数正态分布
plt.subplot(627)
plt.xlabel("random.lognormvariate")
mu = 1
sigma = 0.5
res = [random.lognormvariate(mu, sigma) for _ in range(1,SAMPLE_SIZE)]
plt.hist(res,buckets)

#第八张图 正态分布
plt.subplot(628)
plt.xlabel("random.normalvariate")
mu = 1
sigma = 0.5
res = [random.normalvariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)]
plt.hist(res,buckets)

# 第九张图 帕累托分布
plt.subplot(629)
plt.xlabel("random.paretovariate")
alpha = 1
res = [random.paretovariate(alpha) for _ in range(1, SAMPLE_SIZE)]
plt.hist(res, buckets)

plt.tight_layout()
plt.show()

#真实数据的噪声平滑处理
from pylab import *
from numpy import *
from random import random
def moving_average(interval, window_size): #compute convoluted window for given size
    window = ones(int(window_size)) / float(window_size)
    return convolve(interval,window,'same')
t = linspace(-4,4,100)
y = sin(t) + random(len(t))*0.1 #此处有问题

plt(t,y,"k.")
y_av = moving_average(y,10)
plot(t,y_av,"r")

xlabel("Time")
ylabel("Value")
grid(True)
show()

# Matplotlib

plot([1,2,3,4,5,6]) #y轴点图
plot([4,3,2,1],[1,2,3,4]) #(y,x)轴的值
import matplotlib.pyplot as plt
x = [1,2,3,4] #some data
y = [5,4,3,2]

# create new figure
plt.figure()
plt.subplot(231) # plot折线图
plt.plot(x, y)

plt.subplot(232) # 柱状图
plt.bar(x, y)

plt.subplot(233) # 条状图
plt.barh(x, y)

plt.subplot(234) # stacked bar charts堆叠柱状图
plt.bar(x, y)

y1 = [7,8,5,3]  # more data for stacked bar charts
plt.bar(x, y1, bottom=y, color = 'r') # 底线[5,4,3,2] + 柱长[7,8,5,3]


plt.subplot(235) # 箱线图
plt.boxplot(x)

plt.subplot(236) # 散点图
plt.scatter(x,y)

dataset = [113,115,119,121,124,
           124,125,126,126,126,
           127,127,128,129,130,
           130,131,132,133,136]
plt.subplot(121)
plt.boxplot(dataset, vert=False) #箱线图

plt.subplot(122)
plt.hist(dataset)  #丰度图
plt.show()

# 正弦图和余弦图
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
y = np.cos(x)
y1 = np.sin(x)

plt.plot(x, y)
plt.plot(x, y1)

plt.title("Functions $\sin$ and $\cos$") #标题
plt.xlim(-3.0,3.0) #x轴和y轴的长度限制
plt.ylim(-1.0,1.0)
plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],
           [r'$-\pi$',r'$-\pi/2$',r'$0$',r'$+\pi/2$',r'$+\pi$']) #$-\pi$ 希腊字母
plt.yticks([-1,0,+1],[r'$-1$',r'$0$',r'$+1$']) #设置x和y轴标签
plt.show()

# 坐标轴长度和范围
l = [-1,1,-10,10] #分别表示 xmin,xmax,ymin,ymax
axis(l)


