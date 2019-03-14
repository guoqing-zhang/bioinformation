#encoding=utf-8
'''python学习笔记
搜索
PYTHON 数据可视化实战
PYTHON 机器学习实战
'''

#Numpy
'''
import numpy as np
from numpy.linalg import *
def main():
    lst=[(1,3,5),(2,4,6)]
    print(type(lst))
    np_lst=np.array(lst)
    print(type(np_lst))
    np_lst=np.array(lst,dtype=np.float)
    #bool,int,int8,int16,int32,int64,int128,uint8/16/32/64/128,float16/32/64/128
    print(np_lst.shape)
    print(np_lst.itemsize)
    print(np_lst.ndim)
    print(np_lst.dtype)
    print(np_lst.size)
    #Some arrays
    print(np.zeros([2,4]))
    print(np.ones([2,4]))
    print("Rand:")
    print(np.random.rand(2,4))
    print(np.random.randint(1,10,3))
    print("Randn:")
    print(np.random.randn(1,10,3))
    print(np.random.choice([10,20,30]))
    #Arrays Opes
    print(np.arange(1,11).reshape([2,5]))
    print(np.exp(lst))
    print("Log")
    print(np.log(lst))
    lst1 = np.array([10,20,30,40])
    lst2 = np.array([4,3,2,1])
    print("Add:")
    print(lst1+lst2)
    #add\sub\mul\div\square\dot
    print("Cancatenate")
    #print(np.concatenate(lst1,lst2))

    # liner

    print(np.eye(3))
    #print(np.fft(np.array([1,1,1,1,1,1,1,1,1,1,1,1])))
    print(np.corrcoef([1,0,1],[0,2,1]))
if __name__=="__main__":
    main()
'''
# Figure
'''
import numpy as np

def main():
    #line
    import matplotlib.pyplot as plt
    x=np.linspace(-np.pi,np.pi,256,endpoint=True)
    c,s=np.cos(x),np.sin(x)
    plt.figure(1)
    plt.plot(x,c,color="blue",linewidth=1.0,linestyle="-",label="COS",alpha=0.5)
    plt.plot(x,s,"r*",label="SIN")
    plt.title("COS & SIN")
    ax=plt.gca()
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["left"].set_position(("data",0))
    ax.spines["bottom"].set_position(("data",0))
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],
               [r'$-\pi$',r'$-\pi/2$',0,r'$+\pi/2$',r'$+\pi$'])
    plt.yticks(np.linspace(-1,1,5,endpoint=True))
    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(16)
        label.set_bbox(dict(facecolor="white",edgecolor="None",alpha=0.2))
        plt.legend(loc="upper left")
        plt.grid()
        plt.fill_between(x,np.abs(x)<0.5,c,c>0.5,color="green",alpha=0.25)
        t=1
        plt.plot([t,t],[0,np.cos(t)],"y",linewidth=3,linestyle="--")
        plt.annotate("cos(1)",xy=(t,np.cos(1)),xycoords="data",xytext=(+10,+30),
                     textcoords="offset points",arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2"))
    plt.show()

if __name__ == '__main__':
    main()
'''
#Plot
'''
# scatter
import numpy as np
def main():
    #line
    import matplotlib.pyplot as plt
    fig=plt.figure()
    ax=fig.add_subplot(3,3,1) #3行3列第一个图
    n = 128
    X = np.random.normal(0, 1, n)
    Y = np.random.normal(0, 1, n)
    T = np.arctan2(Y, X)
    #plt.axes([0.025, 0.025, 0.95, 0.95])
    ax.scatter(X, Y, s=75, c=T, alpha=.5)
    plt.xlim(-1.5, 1.5), plt.xticks([])
    plt.ylim(-1.5, 1.5), plt.yticks([])
    plt.axis()
    plt.title("scatter")
    plt.xlabel("x")
    plt.ylabel("y")

    #bar
    fig.add_subplot(3,3,2)
    n = 10
    X = np.arange(n)
    Y1 = (1 - X / float(n) * np.random.uniform(0.5, 1.01, n))
    Y2 = (1 - X / float(n) * np.random.uniform(0.5, 1.01, n))
    plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
    plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')
    for x,y in zip(X,Y1):
        plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')
    for x,y in zip(X,Y2):
        plt.text(x + 0.4, -y - 0.05, '%.2f' % y, ha='center', va='top')


    # Pie
    fig.add_subplot(3, 3, 3)
    n = 20
    Z = np.ones(n)
    Z[-1] *= 2
    plt.pie(Z, explode=Z*.05, colors=['%f' % (i / float(n)) for i in range(n)],
            labels=['%.2f' % (i / float(n)) for i in range(n)])
    plt.gca().set_aspect('equal')
    plt.xticks([]),plt.yticks([])

    #polar
    fig.add_subplot(3, 3, 4, polar=True)
    n = 20
    theta = np.arange(0.0, 2 * np.pi, 2 * np.pi / n)
    radii = 10 * np.random.rand(n)
    plt.polar(theta,radii)
    #plt.plot(theta,radii)

    #heatmap
    fig.add_subplot(3, 3, 5)
    from matplotlib import cm
    data = np.random.rand(3, 3)
    cmap = cm.Blues
    map = plt.imshow(data, interpolation='nearest', cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # 3D
    from mpl_toolkits.mplot3d import Axes3D
    ax=fig.add_subplot(3, 3, 6, projection="3d")
    ax.scatter(1, 1, 3, s=100)

    # hot map
    fig.add_subplot(3, 1, 3)
    def f(x,y):
        return(1 - x / 2 + x ** 5 + y ** 3) * (np.exp(-x ** 2 - y ** 2))
    n = 256
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    X, Y = np.meshgrid(x, y)
    plt.contourf(X, Y, f(X, Y), 8, alpha=.75,cmap=plt.cm.hot)
    #plt.savefig("./data/fig.png")  保存
    plt.show()

if __name__ == '__main__':
    main()
'''
#Integral
'''
import numpy as np
def main():
    # 1--Integral
    from scipy.integrate import quad, dblquad, nquad
    print(quad(lambda x:np.exp(-x), 0, np.inf))
    print(dblquad(lambda t, x: np.exp(-x*t)/t**3,0, np.inf, lambda x: 1, lambda x:np.inf))
    def f(x,y):
        return x*y
    def bound_y():
        return [0,0.5]
    def bound_x(y):
        return [0,1-2*y]
    print(nquad(f,[bound_x,bound_y]))

    #2--Optmizer
    from scipy.optimize import minimize
    def rosen(x):
        return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0+(1-x[:-1])**2.0)
    x0=np.array([1.3,0.7,0.8,1.9,1.2])
    res=minimize(rosen,x0,method="nelder-mead",options={"xtol":1e-8,"disp":True})
    print("ROSE MINI:",res)
if __name__ == '__main__':
    main()

'''
#bar plot
'''
import numpy as np

def main():
    #line
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))

    #x = np.linspace(1, endpoint=True)
    plt.bar(0, 100, width=0.001)
    plt.show()

if __name__ == '__main__':
    main()
'''
#Pandas
'''import pandas as pd
a = pd.Series([1,2,3],index=['a','b','c'])#创建一个序列
b = pd.DataFrame([[1,2,3],[4,5,6]], columns=['a','b','c'])#创建一个表
d2 = pd.DataFrame(a)# 用已有序列创建表格
b.head()#预览前5行数据
b.describe()#数据基本统计量
pd.read_excel('data.xls')#读取excel文件
pd.read_csv('data.csv', encoding='utf-8')#读取文本格式数据，encoding指定编码。
print(d2)'''
#Scikit-Learn /简单鸢尾花线性预测模型
'''from sklearn.linear_model import LinearRegression #导入线性回归模型
model = LinearRegression()#建立线性回归模型
from sklearn import datasets#导入数据集

iris = datasets.load_iris()#加载数据集
#print(iris.data.shape)#查看数据集大小
from sklearn import svm

clf = svm.LinearSVC()#建立线性SVM分类器
clf.fit(iris.data,iris.target)#使用数据训练模型
clf.predict([[5.0,3.6,1.3,0.25]])#输入新的数据进行预测
#print(clf.coef_)#查看训练好模型的参数
'''
#相关性计算
'''
import pandas as pd
D = pd.DataFrame([range(1,8),range(2,9)])
D.corr(method='spearman')
S1 = D.loc[0]
S2 = D.loc[1]
print(S1)
print(S2)
S1.corr(S2,method='pearson')
print(D.corr(method='spearman'))
print(S1.corr(S2,method='pearson'))
'''
#作图
'''import matplotlib.pyplot as plt
import numpy as np
#x = np.linspace(0,2*np.pi,50)# x坐标输入
#y = np.sin(x)# 计算对应x的正弦值
#plt.plot(x,y,'r--')
#plt.show()
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs' #定义标签
sizes = [15, 30, 45, 10] #每一块的比例
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']#每一块的颜色
explode = (0, 0.1 ,0 ,0) #突出显示
plt.pie(sizes, explode=explode, labels=labels,colors=colors,
        autopct='%1.1f%%',shadow=True,startangle=90)
plt.title('Pie') #标题
plt.axis('equal')#显示为圆，避免比例压缩为椭圆
plt.show()'''
#拉格朗日插补空值
'''
import pandas as pd
from scipy.interpolate import lagrange#导入拉格朗日插值函数

inputfile='../data/catering_sale.xls'#输入路径
outputfile='sales.xls'#输出路径

data=pd.read_excel(inputfile)
data[u'销量'][(data[u'销量']<400) | (data[u'销量']>5000)] = None#过滤异常值，将其变为空值
#自定义列向量插值函数
#s为列向量，n为被插值的位置，k为取前后的数据个数，默认为5
def ployinterp_column(s,n,k=5):
    y=s[list(range(n-k,n))+list(range(n+1,n+1+k))]#取数
    y = y[y.notnull()]#剔除空值
    return lagrange(y.index,list(y))(n)#插值并返回插值结果
#逐个元素判断是否需要插值
for i in data.columns:
    for j in range(len(data)):
        if (data[i].isnull())[j]:#如果为空即插值
            data[i][j]=ployinterp_column(data[i],j)
data.to_excel(outputfile)#输出结果，写入文件
x=0
'''
#决策树算法预测销量高低
#-*- coding: utf-8 -*-
#使用ID3决策树算法预测销量高低
import pandas as pd

#参数初始化
'''
filename = '../python_code/chapter5/chapter5/demo/data/sales_data.xls'

def ask_ok(prompt, retries = 4, complaint='Yes or no, please!'):
    while True:
        ok = input(prompt)
        if ok in ('y', 'ye', 'yes'):
            return True
        if ok in ('n', 'no', 'nop', 'nope'):
            return False

        retries = retries - 1
        if retries < 0:
            raise OSError('uncooperative user')
        print(complaint)

def f(a, L=None):
    if L is None:
        L = []
    L.append(a)
    return L

'''
# 作用域和命名空间示例
'''
def scope_test():
    def do_local():
        spam = "local spam"
    def do_nonlocal():
        nonlocal spam
        spam = "nonlocal spam"
    def do_global():
        global spam
        spam = "global spam"
    spam = "test spam"
    do_local()
    print("After local assignment:", spam)
    do_nonlocal()
    print("After nonlocal assignment:", spam)
    do_global()
    print("After global assignment:", spam)

scope_test()
print("In global scope:", spam)
'''
# 类对象
'''
class MyClass:
    """A simple example class"""
    i = 12345
    def f(self):
        return 'hello world'
class Complex:
    def __init__(self,realpart,imagpart):
        self.r = realpart
        self.i = imagpart

x = Complex(3.0, -4.5)
x.r, x.i

class Dog:
    kind = 'canine' # class variable shared by all instances
    def __init__(self, name):
        self.name = name # instance variable unique to each instance

    class Bag:
        def __init__(self):
            self.data = []

    def add(self, x):
        self.data.append(x)

    def addtwice(self, x):
        self.add(x)
        self.add(x)
'''
#名称重整有助于子类重写方法，而不会打破组内的方法调用。
'''
class Mapping:
    def __init__(self, iterable):
        self.items_list = []
        self.__update(iterable)
    def update(self, iterable):
        for item in iterable:
            self.items_list.append(item)
    __update = update # private copy of original update() method

# 迭代器
class Reverse:
    """Iterator for looping over a sequence backwards."""
    def __init__(self, data):
        self.data = data
        self.index = len(data)
    def __iter__(self):
        return self
    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.data[self.index]

class MappingSubclass(Mapping):
    def update(self, keys, values):
        # provides new signature for update()
        # but does not break __init__()
        for item in zip(keys, values):
            self.items_list.append(item)
'''
# U-test
'''
from scipy.stats import mannwhitneyu
from scipy.stats import ranksums
import pandas as pd
import numpy as np
a= np.array([57.07168,46.95301,31.86423,38.27486,77.89309,76.78879,33.29809,58.61569,18.26473,62.92256,50.46951,19.14473,22.58552,24.14309])
b= np.array([8.319966,2.569211,1.306941,8.450002,1.624244,1.887139,1.376355,2.521150,5.940253,1.458392,3.257468,1.574528,2.338976])

Statistic,Pvalue = mannwhitneyu(a,b)
print("Statistic is:",Statistic)
print("Pvalue is:",Pvalue)
print('-'*40)
Statistic,Pvalue = ranksums(a,b)
print("Statistic is:",Statistic)
print("Pvalue is:",Pvalue)
'''

# 标准库
'''
import os #与操作系统交互
import glob #从目录通配符搜索中生成文件列表
import sys #命令行参数
import re #高级字符串正则表达式工具
import math #浮点数运算
import random #生成随机数
#Scipy 提供数值计算模块
from datetime import date #日期和时间处理
import zlib #支持通用数据打包格式 zlib,gzip,bz2,lzma,zipfile,tarfile
from timeit import Timer #性能度量 profile 和 pstats 针对更大代码块
Timer('t=a; a=b; b=t', 'a=1; b=2').timeit()
Timer('a,b = b,a', 'a=1; b=2').timeit()
'''

# 文件处理
import numpy as np
import pandas as pd
Reverse_table = pd.read_csv("Quality/reverse-seven-number-summaries.csv")
Reverse_list = np.array(Reverse_table[4:5]).tolist()[0]
Reverse_list.remove('50%')
n = 0
for i in Reverse_list:
    n = n+1
    if i < 20:
        print(n-1) #小于20的前一位
        break
Forward_table = pd.read_csv("Quality/forward-seven-number-summaries.csv")
Forward_list = np.array(Forward_table[4:5]).tolist()[0]
Forward_list.remove('50%')
n = 0
for i in Forward_list:
    n = n+1
    if i < 20:
        print(n-1) #小于20的前一位
        break
    while n == len(Forward_list):
        print(n)
        break
# CSV I/O
import csv
with open('books.csv') as rf:
    reader = csv.reader(rf)
    headers = next(reader)
    with open('book_out.csv', 'w') as wf:
        writer = csv.writer(wf)
        writer.writerrow(headers)

        for book in reader:
            price = book[-2]
            if price and float(price) >= 80.00:
                writer.writerrow(headers)

#过滤列表
from random import randint
l = [randint(-10, 10) for _ in range(10)]
[x for x in l if x >= 0] #第一种方法，列表解析
g = filter(lambda x: x>=0,l) #第二种方法，返回一个生成器，next(g)
list(g)
#过滤字典
d = {'student%d' %i: randint(50, 100) for i in range(1, 21)} #随机生成分数，20人
g = {k:v for k, v in d.items() if v >= 90} #分数高于90
g = filter(lambda item:item[1]>=90,d.items()) #filter方法
dict(g) #构造字典
#过滤集合
s = {randint(0,20) for _ in range(20)}
s1 = {x for x in s if x % 3 == 0}

#定义数值常量
NAME = 0
AGE = 1
SEX = 2
EMAIL = 3
NAME,AGE,SEX,EMAIL = range(0,4) #简洁方式,元组拆包
#枚举
from enum import IntEnum
class StudentEnum(IntEnum):
    NAME = 0
    AGE = 1
    SEX = 2
    EMAIL = 3

s[StudentEnum.NAME]  #第一种方法
from collections import namedtuple
Student = namedtuple('Student',['name','age','sex','email']) #第二种方法,命名元组
s2 = Student('Jim',16,'male','Jim8721@gmail.com')
s2.age
s2.name

#字典排序 sorted可对元组排序
from random import randint
d = {k: randint(60, 100) for k in 'abcdefgh'}
l = [(v, k)for k, v in d.items()]
sorted(l, reverse=True) #排序，reverse方向
list(zip(d.values(), d.keys()))
p = sorted(d.items(), key=lambda item: item[1], reverse=True)
list(enumerate(p, 1)) #产生次序

for i,(k,v) in enumerate(p):
    print(i,k,v)
    d[k] = (i,v)
d = {k: (i, v) for i,(k, v) in enumerate(p)}

# 元素出现频度
from random import randint
data = [randint(0, 20) for _ in range(30)]
d = dict.fromkeys(data, 0)
for x in data:
    d[x] += 1
sorted([(v, k) for k, v in d.items()], reverse=True)[:3] #列表解析
sorted(((v, k) for k, v in d.items()), reverse=True)[:3] #生成器解析

import heapq
heapq.nlargest(3, ((v, k)for k,v in d.items()))

from collections import Counter
c = Counter(data)
c.most_common(3)
import re #字符频率
txt = open('test.txt').read()
word_list = re.split('\w+',txt)
c2 = Counter(word_list)
c2.most_common(10)

# 多个字典的公共键
from random import randint,sample
sample('abcdefgh', 3)
sample('abcdefgh', randint(3, 6))
d1 = {k:randint(1, 4) for k in sample('abcdefgh', randint(3, 6))}
d2 = {k:randint(1, 4) for k in sample('abcdefgh', randint(3, 6))}
d3 = {k:randint(1, 4) for k in sample('abcdefgh', randint(3, 6))}
[k for k in d1 if k in d2 and k in d3] #简单方法
[k for k in d1[0] if all(map(lambda d: k in d,d1[1:]))]
s1 = d1.keys() #利用集合(set)的交集操作
s2 = d2.keys()
dl = [d1,d2,d3]
from functools import reduce
reduce(lambda a, b: a*b, range(1, 11))
reduce(lambda a,b:a&b, map(dict.keys(),dl))

# 如何让字典保持有序
d = {}
d['c'] = 1
d['b'] = 2
d['a'] = 3
d.keys()
from collections import OrderedDict
od = OrderedDict()
od['c'] = 1
od['b'] = 2
od['a'] = 3
od.keys()
players = list('abcdefgh')
from random import shuffle
shuffle(players)
od = OrderedDict()
for i,p in enumerate(players, 1):
    od[p] = i

def query_by_name(d, name):
    return d[name]

from itertools import islice
islice(range(10),3,6) #迭代

def query_by_order(d,a,b=None):
    a-=1
    if b is None:
        b= a + 1
    return list((islice(od,a,b)))
query_by_order(od, 4)
query_by_order(od, 3, 4)

# 实现用户历史纪录功能
from collections import deque #第一种方法
q = deque([],5)
q.append(1)
q.append(2)
q.append(3)
q.append(4)
q.append(5)


from random import randint
from collections import deque

def guess(n, k): #猜数字游戏
    if n == k:
        print('猜对了！这个数字是%d。' % k)
        return True
    if n < k:
        print('猜大了，比%d小。' % k)
    elif n > k:
        print('猜小了，比%d小。' % k)
    return False

def main():
    n = randint(1, 100)
    i = 1
    hq = deque([], 5)
    while True:
        line = input('[%d] 请输入一个数字：' % i)
        if line.isdigit():
            k = int(line)
            hq.append(k)
            i += 1
            if guess(n,k):
                break
        elif line == 'quit':
            break
        elif line == 'h?':
            print(list(hq))
if __name__ == '__main__':
    main()

import pickle #第二种方法。pickle.dump 和 pickle.load

#在列表，字典，集合中根据条件筛选数据
from random import randint
data = [randint(-10,10) for _ in range(10)]
filter(lambda x: x >= 0, data) #第一种方式，filter函数
[x for x in data if x >= 0] #第二种方式，列表解析

d = {x: randint(60, 100) for x in range(1,21)}
d1 = {k:v for k,v in d.items() if v > 90}
s=set(data)
s1 = {x for x in s if x %3 == 0}

# 拆分含有多种分隔符的字符串
s = 'ab;cd|efg|hi,jkl|mn\topq;rst,uvw\txyz' #第一种方法,多次使用str.split()
s.split(';')
[ss.split('|') for ss in s.split(';')] #split方法
list(map(lambda ss:ss.split('|'),s.split(';'))) #列表解析方法
t=[]
map(t.extend,[ss.split('|') for ss in s.split(';')])

weight,height = eval(input("Please input weight and height:"))
bmi = weight/height ** 2
print("Your BMI is {0:.1f}".format(bmi))
if bmi < 18.5:
    print("too thin")
elif bmi < 24:
    print("normal")
elif bmi < 27.9:
    print("overweight")
else:
    print("fat")
# 华氏度转摄氏度
for f in range(0,301,20):
    c = 5/9 * (f - 32)
    print("{} f = {:.0f} c".format(f,c))
'''
from numpy import *
CSGroups = mat([0.00872068,0.00436034,0.1831342,0.0436034,0,0.2267376,0.00436034,
                0,0,16.0591262,0.0959274,0.013081,0.0479637,0.1395308,1.4476323,0,
                0,0,0.2223773,7.0157844,0.6278887,0.00436034,0.0741258,0.1482515,
                0.0566844,0.00872068,0.00436034,0.00436034,0.00872068,0.00436034,
                0.00436034,0.0872068,0.0784861,0.4709165,0.1002878,0,0,1.3517049,
                0.7717799,3.4969914,0,1.5740821,0.1438912,0.4360338,0.4534752,0.4621959,
                0,0,3.8196564,3.7673323,0.7150955,0.5973664,0,1.4607134,0,0.00436034,
                0.3706288,0.0915671,5.1495596,0.0174414,0.837185,0.1918549,0.8110229,
                0,0.00872068,0,0.0174414,0,2.049359,0,0.0523241,0.606087,0,0.0218017,
                0.00436034,0.0523241,0.0305224,0.1220895,0,0.00436034,0.1177291,0.013081,
                0,0,0.0784861])

AS3 = mat([0,0,0.05232406,0.004360338,0.135170489,0,0,0.004360338,50.13081015,0.05232406,
           0.004360338,0.034882707,0.470916543,0.008720677,0.008720677,0.02616203,
           0.043603384,0.122089474,0.078486091,0,0,0,0,0,0,0,0,0,0,0,0,0.013081015,
           0.183134211,0,0,0.218016918,0.431673498,1.630766547,0.148251504,0.091567106,
           0.715095491,0.287782332,0.017441353,0,2.489753205,0,1.678730269,0.305223685,
           13.98796547,8.75991977,0,0,0.422952821,0,7.477980291,0.004360338,0.043603384,
           0,0.087206767,0.004360338,0.082846429,0,0,0.017441353,0.043603384,0.174413534,
           0,0.095927444,0,4.07255603,0.004360338,0,0.23981861,0.013081015,0.008720677,
           0.274701317,0,0,1.796459405,2.58132031,0,0,0.370628761,0.047963722,0])
print(sqrt((CSGroups -AS3)*((CSGroups-AS3).T)))

from numpy import *
MetaCSGroups = mat([6633.86,118.78,7.31,6479.94,55.21,312.95,8.56,2592.05,6.8,47.3,
                    161.25,2357.43,50.08,4.53,30.38,52.06,323.11,78.66,17057.94,
                    43.42,92.41,47.57,2.21,2696.4,32.88,30.9,254.57,10.33,234.29,
                    2936.67,30.81,25.04,19.94,1.38,35.84,10.01,48.93,379.68,311.33,
                    21.62,81.8,16.92,104.63,83.21,49.03,43701.63,154.44,1455.71,
                    185.4,3141.13,2603.3,207.48,179.63,44.6,4.47,50.8,18244.84,
                    399.06,155.67,221.45,14244.15,125.82,27.07,103.77,86.55,77.74,
                    1230.3,221.86,7593.18,327.25,644.78,3626.23,666.63,1340.82,
                    4244.06,4268.5,216.57,497.51,644.81,541.61,188.75,1030.41,
                    943.55,414.51,16.5,77.64,47.66,562.09,77.53,308.12,1736.73,
                    135.64,12.34,43.38,11.5,185.24,2001.06,1013.79,1926.21,153.43,
                    19577.92,22.98,198.88,547.13,7.91,175.9,181.01,5643.6,33.99,
                    149.46,1113.12,392.98,42.3,236.1,47.82,17.74,402810.64,33.35])

MetaAS3 = mat([15018.59,66.07,45.95,39668.07,12.05,31.51,10.15,480.5,2.89,29.31,45.57,
               710.53,40.59,39.52,2.4,97,155.42,15.14,43512.61,606.24,104.37,210.44,1.77,
               670.02,10.97,3.51,134.24,5.44,81.87,115.08,643.49,0.69,55.27,101.97,77.08,
               12.41,27.92,278.76,2263.24,17.9,111.37,9.09,581.28,104.56,432.86,1797.86,
               42.61,1694.94,29.88,654.26,683.63,704.31,114.83,8.26,0.33,29.61,1610.36,
               8.76,157.06,30.68,507.86,61.83,11.66,352.06,46.83,25.97,401.22,989.24,
               1115.61,105.14,1193.13,839.75,657.86,1358.89,1225.34,338.95,319.79,
               468.59,303.66,143.86,118.74,580.59,1023.72,176.16,6.49,38.91,9.71,
               196.89,72.1,173.63,981.6,9.39,3.58,853.55,0.02,2.21,2552.46,2221.13,
               671.28,119.75,89.71,6.37,10.78,14.94,21.93,45.76,207.73,841.07,186.81,
               10.91,108.84,343.36,13.54,1194.83,42.53,8.74,9958.74,16.7])
e = sqrt((MetaCSGroups -MetaAS3)*((MetaCSGroups-MetaAS3).T))
'''

import numpy as np
a = ([53.06,
54.38,
51.81,
53.21,
53.2,
54.1,
52.81,
53.79,
52.72,
53.81,
52.98,
53.38,
52.7,
54.26,
52.73,
53.98,
52.29,
53.52,
52.88,
53.94,
52.92,
54.01,
52.41,
53.73,
52.74,
53.9,
51.74,
53.43,
51.94,
53.21,
51.28,
53.14,
51.4,
52.2,
52.01,
54.25,
51.76,
53.09,
52.08,
53.75,
51.45,
53.28,
52.14,
54.29,
52.23,
54.13,
53.11,
54.58,
52.1,
53.15,
52.32,
53.81,
52.03,
52.77,
52.41,
54.06,
52.18,
53.36,
52.94,
54.42,
51.65,
53.26,
52.62,
54.13,
52.51,
54.45,
52.59,
54.21,
52.07,
54.1,
52.07,
53.44,])
np.std(a)
# matplotlib 画图
# sphinx_gallery_thumbnail_number = 3
import matplotlib.pyplot as plt
import numpy as np
import pandas
fig = plt.figure() # an empty figure with no axes
fig.suptitle('No axes on this figure') # Add a title so we know which it is
fig,ax_lst = plt.subplot(2, 2) # a figure with a 2x2 grid of Axes
a = pandas.DataFrame(np.random.rand(4,5),columns=list('abcde'))#创建一个4*5的矩阵
a_asndarray= a.values #获得a的值为数组
b = np.matrix([[1,2],[3,4]]) # 2*2的数组
b_asarray = np.asarray(b) #获得b的值为数组

x= np.linspace(0,2,100)
plt.plot(x,x,label='linear')
plt.plot(x,x**2,label='quadratic')
plt.plot(x,x**3,label='cubic')
plt.xlabel('x label')
plt.ylabel('y label')
plt.title("Simple plot")
plt.legend() #显示数据标签
plt.show()

import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0,10,0.2)
y = np.sin(x)
fig, ax =plt.subplots()
"""
fig,ax = plt.subplots()的意思是，同时在subplots里建立一个fig对象，建立一个axis对象 
这样就不用先plt.figure() 
再plt.add_subplot()了
"""
ax.plot(x,y)
plt.show()

def my_plotter(ax,data1,data2,param_dict):
    """
    A helper function to make a graph
    Parameters
    ----------
    ax : Axes
        The axes to draw to

    data1 : array
       The x data

    data2 : array
       The y data

    param_dict : dict
       Dictionary of kwargs to pass to ax.plot

    Returns
    -------
    out : list
        list of artists added
    """
    out = ax.plot(data1,data2, **param_dict)
    return out
# which you would then use as:

data1, data2, data3, data4 = np.random.randn(4, 100)
fig, ax = plt.subplots(1, 1)
my_plotter(ax, data1, data2, {'marker': 'x'})
# 两个子图
fig, (ax1, ax2) = plt.subplots(1, 2)
my_plotter(ax1, data1, data2, {'marker': 'x'})
my_plotter(ax2, data3, data4, {'marker': 'o'})

# pyplot教程
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
"""如果为plot()命令提供单个列表或数组 ，matplotlib假定它是一系列y值，
并自动为您生成x值。由于python范围以0开头，因此默认的x向量与y具有相同的长度，
但从0开始。因此x数据为 [0,1,2,3]。"""
plt.show()

plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro') #第三个参数，格式字符串，指示绘图的颜色和线型。

plt.axis([0, 6, 0, 20]) #x轴 0-6 y轴0-20 [xmin, xmax, ymin, ymax]
plt.show()

import numpy as np
# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)
# red dashes,blue squares and green triangles
plt.plot(t,t,'r--',t,t**2,'bs',t,t**3,'g^')
plt.show()
# 关键字符串绘图
data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

plt.scatter('a', 'b', c='c', s='d', data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.show()
# 分类变量绘图
names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]

plt.figure(1, figsize=(9, 3))

plt.subplot(131) #1行3列第1图
plt.bar(names, values) #列表名和值对应
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
plt.show()

# 线的属性
plt.plot(x,y,linewidth=2.0) #线宽
line, = plt.plot(x, y, '-') #‘-’线的形状
line.set_antialiased(False) # turn off antialising
"""
lines = plt.plot(x1, y1, x2, y2)
# use keyword args
plt.setp(lines, color='r', linewidth=2.0)
# or MATLAB style string value pairs
plt.setp(lines, 'color', 'r', 'linewidth', 2.0)
"""

#多个图形和轴
def f(t):
    return np.exp(-t)*np.cos(2*np.pi*t)
t1 = np.arange(0.0,5.0,0.1)
t2 = np.arange(0.0,5.0,0.02)

plt.figure(1)
plt.subplot(211)
plt.plot(t1,f(t1),'bo',t2,f(t2),'k')

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()

ax = plt.subplot(111)

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)

plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             ) #指定标签位置和箭头位置

plt.ylim(-2, 2) #y轴限度
plt.show()

# 对数和其他非线性轴
from matplotlib.ticker import NullFormatter  # useful for `logit` scale

# Fixing random state for reproducibility
np.random.seed(19680801)

# make up some data in the interval ]0, 1[
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
y = y[(y > 0) & (y < 1)]
y.sort()
x = np.arange(len(y))

# plot with various axes scales
plt.figure(1)

# linear
plt.subplot(221)
plt.plot(x, y)
plt.yscale('linear')
plt.title('linear')
plt.grid(True)


# log
plt.subplot(222)
plt.plot(x, y)
plt.yscale('log')
plt.title('log')
plt.grid(True)


# symmetric log
plt.subplot(223)
plt.plot(x, y - y.mean())
plt.yscale('symlog', linthreshy=0.01)
plt.title('symlog')
plt.grid(True)

# logit
plt.subplot(224)
plt.plot(x, y)
plt.yscale('logit')
plt.title('logit')
plt.grid(True)
# Format the minor tick labels of the y-axis into empty strings with
# `NullFormatter`, to avoid cumbering the axis with too many labels.
plt.gca().yaxis.set_minor_formatter(NullFormatter()) #格式美化
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35) #调整顶部，底部，左，右，水平间距，垂直间距

plt.show()

#列表解析
L = [1,2,3,4,5]
L = [x + 10 for x in L]
#扩展的列表解析方法
lines = [line.rstrip() for line in open('script1.py') if line[0] == 'p']
'''if子句检查从文件读取的每一行，看它的第一个字符是否是p;如果不是，从结果列表中省略该行。'''
[x + y for x in 'abc' for y in 'lmn']

#函数调用中特殊的*arg形式，把一个集合的值解包为单个的参数。
def f(a,b,c,d):print(a,b,c,d,sep='&')
#zip内置函数可以把zip过来的元组unzip
X = (1,2)
Y = (3,4)
list(zip(X,Y))
A,B = zip(*zip(X,Y))

#字典迭代
D = dict(a=1,b=2,c=3)

K = D.keys()
I = iter(K)
next(I)
for k in D.keys():print(k,end=' ')

# global声明一个模块级的变量并被赋值。
# 函数中使用global语句声明，可以在整个模块中都可以使用的变量名。
# nonlocal 声明将要赋值的一个封闭的函数变量。

def times(x, y):
    return x * y
# 搜索交集的工具
def intersect(seq1, seq2):
    res = [] # 被赋值，所以是典型的本地变量
    for x in seq1:
        if x in seq2:
            res.append(x)
    return res
s1,s2 = ('SPAM','SCAM')
intersect(s1,s2) # 调用
[x for x in s1 if x in s2] #列表解析表达式，取交集

# 作用域实例
# Global scope
X = 99                # X and func assigned in module:global
def func(Y):          # Y and Z assigned in function:locals
    #Lacal scope
    Z = X + Y         #  X is a global
    return Z

func(1)               # func in module:result=100
'''
全局变量名：X，func   因为X是在模块文件顶层注册的，所以是全局变量。def语句在 模块文件顶层将一个函数对象
赋值给了变量名func,因此func也是全局变量。
本地变量名：Y，Z      Y和Z是本地变量（并且只在函数运行时存在），因为他们都是在函数定义内部进行赋值的，Z
通过=语句。
变量名隔离机制，函数内变量名不会与模块命名空间内的变量名产生冲突。
'''
X = 88
def func():
    X = 99  # 不增加global(或nonlocal)声明的话，是没有办法在函数内改变函数外部的变量的。
    # global X；X = 99
func()
print(X) #结果为88；函数内部的赋值语句创建了本地变量X，与函数外部模块文件中的全局变量X是完全不同的变量。

# global语句；声明命名空间
y, z = 1, 2
def all_global():
    global x
    x = y + z
# 在不熟悉编程的情况下，最好尽可能地避免使用全局变量。
# 在文件间进行通信最好的办法就是通过调用函数，传递参数，然后得到其返回值。
# first.py
X = 99
def setX(new):
    global X
    X = new
# second.py
#import first
#first.setX(88)

# 工厂函数 / 作用域
def maker(N):
    def action(X):
        return X ** N
    return action

def f1():
    x = 88 # arg = val: 参数arg在调用时没有值传入进来的时候，默认会使用值val
    def f2(x=x):
        print(x)
    f2()
f1()

def func():
    x = 4
    action = (lambda n: x ** n)
    return action
x = func()
print(x(2))

def tester(start):
    state = start
    def nested(label):
        nonlocal state # 声明为nonlocal
        print(label, state)
        state += 1
    return nested

# 传递参数
# 参数的传递是通过自动将对象赋值给本地变量名来实现的。
# 在函数内部的参数名的赋值不会影响调用者。
# 改变函数的可变对象参数的值也许会对调用者有影响。
# func(value)      常规参数；通过位置进行匹配
# func(name=value) 关键字参数；通过变量名匹配
# func(*sequence)  以name传递所有的对象，并作为独立的基于位置的参数
# func(**dict)     以name成对的传递所有的关键字/值，并作为独立的关键字参数
# def func(name)   常规参数；通过位置或变量名进行匹配
# def func(name=value) 默认参数值，如果没有在调用中传递的话
# def func(*name)  匹配并收集（在元组中）所有包含位置的参数
# def func(**name) 匹配并收集（在字典中）所有办好位置的参数
# def func(*args, name)  参数必需在调用中按照关键字传递
# def func(*, name=value)
def func(spam, eggs, toast=0, ham=0):
    print((spam, eggs, toast, ham))

func(1,2)                     #Output:(1,2,0,0)
func(1,ham=1,eggs=2)          #Output:(1,0,0,1)
func(spam=1,eggs=2)           #Output:(1,0,0,0)
func(toast=1,eggs=2,spam=3)   #Output:(3,2,1,0)
func(1,2,3,4)                 #Output:(1,2,3,4)
# */**在函数头部意味着收集任意数量的参数；在调用时，解包任意数量的参数。

def kwonly(a, *, b, c): # *号后都为关键字参数
    print(a, b, c)
def func(x, y, z):
    return x + y + x
f = lambda x, y, z: x + y + z
# 迭代
res = []
for x in 'spam':
    res.append(ord(x))
# 等同于
res = list(map(ord, 'spam')) # map把一个函数映射遍一个系列
res = [ord(x) for x in 'spam'] # 列表解析把一个表达式映射遍一个序列
res = list(filter((lambda x: x % 2 ==0),range(5)))

#列表解析和矩阵
M = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]]

M[1] #行
M[1][2] #单个值
[raw[1] for raw in M] #迭代，获得列
[M[i][i] for i in range(len(M))] #偏移量获得对角线

#生成器
def gensquares(N):
    for i in range(N):
        yield i ** 2




##
## PYTHON 数据可视化编程实战
##
#

# 示例代码路径
PATH = r'D:/pythondemo/python数据可视化编程实战/PythonDataVisualization/Chapter 02/'

#读取CSV文件/Code有错误/大概版本问题
import csv
import sys
import numpy

data = []
try:
    with open(PATH+'ch02-data.csv','r') as f:
        reader = csv.reader(f)
    header = reader.next()
    data = [row for row in reader]
except csv.Error as e:
    print('Error reading CSV file at line %s:%s' %(reader.line_num, e))
    sys.exit(-1)
if header:
    print(header)
    print("=================")
for datarow in data:
    print(datarow)

# 另一种方式/加载大文件
import numpy
data = numpy.loadtxt(PATH+'ch02-data.csv',dtype='str',delimiter=',')

# 导入EXCEL文件
import xlrd
file = 'ch02-xlsxdata.xlsx'
wb = xlrd.open_workbook(filename=PATH+file)
ws = wb.sheet_by_name('Sheet1')
dataset = []
for r in range(ws.nrows):
    col = []
    for c in range(ws.ncols):
        col.append(ws.cell(r,c).value)
    dataset.append(col)
print(dataset)

# 从定宽数据文件导入数据
import struct
import string
datafile = 'ch02-fixed-width-1M.data'
mask = '9s14s5s'
with open(PATH+datafile, 'r') as f:
    for line in f:
        fields = struct.Struct(mask).unpack_from(line)
        print('fields: ',[field.strip() for field in fields])

# 从JSON数据源导入数据

import requests

url = 'https://github.com/timeline.json' #指定GitHub URL

r = requests.get(url)
json_obj = r.json()

repos = set()
for entry in json_obj:
    try:
        repos.add(entry['repository']['url'])
    except KeyError as e:
        print('No key %s. Skipping...' % (e))
print(repos)

from matplotlib.pyplot import *

# 设置坐标轴长度和范围
matplotlib.pyplot.axis()
# 网格线
matplotlib.pyplot.grid() #which指定绘制的网格刻度类型（major minor both）axis 指定绘制哪组网格线（both x y）

def f(x=input()):
    return x**2
# 定义类
class Rectangle():
    def __init__(self, w, l):
        self.width = w
        self.len = l
    def area(self):
        return self.width * self.len
    def change_size(self, w, l): # 定义多种方法
        self.width = w
        self.len = l

# 面向对象编程有四大概念：封装、抽象、多态和继承。
# 继承
class Shape(): #父类
    def __init__(self, w, l):
        self.width = w
        self.len = l
    def print_size(self):
        print("""{} by {}
        """.format(self.width,self.len))

class Square(Shape): #子类
    def area(self):
        return self.width * self.len #可以定义新的方法和变量。
# 后者继承了Shape类的变量和方法。
a_square = Square(20,20)
a_square.print_size()

def main():
    pass
if __name__ == '__main__':
    main()
'''
定义类
创建一个实例
通过实例使用属性或方法
'''
class Dog():
    counter = 0 #类属性，静态属性
    def __init__(self,name):
        self.name = name

class roster(object):
    "students and teacher class"
    teacher = ""
    students = [ ]
    def __init__(self,tn = 'Niuyun'):
        self.teacher = tn
    def add(self, sn):
        self.students.append(sn)
    def remove(self, sn):
        self.students.remove(sn)
    def print_all(self):
        print("Teacher:", self.teacher)
        print("Students:",self.students)




##
##PYTHON 机器学习实战
##

# 导入必要模块
import os
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier #导入随机森林分类器
from sklearn.model_selection import train_test_split #将数据分成训练组和测试组的模块
# 指定路径并下载数据集
PATH = r'D:/pythondemo/'
r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
with open(PATH + 'iris.data', 'w') as f:
    f.write(r.text)
# 读取数据集并展示前5行
df = pd.read_csv(PATH+'iris.data',names=['sepal length','sepal width','petal length','petal width','class'])
df.head() # 前5行
virginica = df[df['class']=='Iris-virginica'].reset_index(drop=True) #提取符合要求的行，并重置index

# 可视化
import matplotlib.pyplot as plt
plt.style.use('ggplot') #设置风格为近似R中的ggplot
import numpy as np
fig, ax = plt.subplots(figsize=(6,4)) #创建宽度6英寸，高度4英寸的插图。
ax.hist(df['petal width'], color='black') #hist传入数据，直方图颜色设置为black
ax.set_ylabel('Count', fontsize=12) #设置y轴标签和大小
ax.set_xlabel('Width', fontsize=12) #设置x轴标签和大小
plt.title('Iris Petal Width' , fontsize=14, y=1.01) #设置标题，y调整了标题在y轴方向相对于图片顶部的位置。

fig, ax = plt.subplots(2,2, figsize=(6,4)) # 2*2四个子图
ax[0][0].hist(df['petal width'], color = 'black')
ax[0][0].set_ylabel('Count', fontsize=12)
ax[0][0].set_xlabel('Widrh', fontsize=12)
ax[0][0].set_title('Iris Petal Width', fontsize=14, y=1.01)

ax[0][1].hist(df['petal length'], color = 'black')
ax[0][1].set_ylabel('Count', fontsize=12)
ax[0][1].set_xlabel('Length', fontsize=12)
ax[0][1].set_title('Iris Petal Length', fontsize=14, y=1.01)

ax[1][0].hist(df['sepal width'], color = 'black')
ax[1][0].set_ylabel('Count', fontsize=12)
ax[1][0].set_xlabel('Widrh', fontsize=12)
ax[1][0].set_title('Iris Sepal Width', fontsize=14, y=1.01)

ax[1][1].hist(df['sepal length'], color = 'black')
ax[1][1].set_ylabel('Count', fontsize=12)
ax[1][1].set_xlabel('Length', fontsize=12)
ax[1][1].set_title('Iris Sepal Length', fontsize=14, y=1.01)
plt.savefig(PATH + 'Iris.pdf')  #保存为pdf格式
# 散点图
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(df['petal width'],df['petal length'], color='green')
ax.set_xlabel('petal width')
ax.set_ylabel('petal length')
ax.set_title('Petal Scatterplot')
# 折线图
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(df['petal length'], color='blue')
ax.set_xlabel('Specimen Number')
ax.set_ylabel('petal length')
ax.set_title('Petal Length Plot')
# 堆叠条形图
fig, ax = plt.subplots(figsize=(6,6))
bar_width = .8
labels = [x for x in df.columns if 'length' in x or 'width' in x] #带有length和width的标签index列表
ver_y = [df[df['class']=='Iris-versicolor'][x].mean() for x in labels] #各列的平均值
vir_y = [df[df['class']=='Iris-virginica'][x].mean() for x in labels]
set_y = [df[df['class']=='Iris-setosa'][x].mean() for x in labels]
x = np.arange(len(labels))
ax.bar(x, vir_y, bar_width, bottom=set_y,color='darkgrey') #bottom参数需要传入当前类别-1的数据总和。
# bottom这个参数将该序列的y点最小值设置为其下面那个序列的y点最大值。
ax.bar(x, set_y, bar_width, bottom=ver_y,color='white')
ax.bar(x, ver_y, bar_width, color='black')
ax.set_xticks(x + (bar_width/2)) #设置标签之间的间隔
ax.set_xticklabels(labels, rotation=-70, fontsize=12) #传入想显示的列名
ax.set_title('Mean Feature Measurement By Class', y=1.01)
ax.legend(['Virginica','Setosa','Versicolor']) #定义标签

# Seaborn库
import seaborn as sns
sns.pairplot(df, hue='class') # 所有特征图
plt.savefig(PATH+'Seaborn.pdf') #保存图片

# matplotlib可以修改并使用seaborn / 小提琴图
fig,ax = plt.subplots(2, 2, figsize=(7, 7))
sns.set(style='white', palette='muted')
sns.violinplot(x=df['class'], y=df['sepal length'], ax=ax[0,0])
sns.violinplot(x=df['class'], y=df['sepal width'], ax=ax[0,1])
sns.violinplot(x=df['class'], y=df['petal length'], ax=ax[1,0])
sns.violinplot(x=df['class'], y=df['petal width'], ax=ax[1,1])
fig.suptitle('Violin Plots', fontsize=16, y=1.03) #在所有的子图上添加一个总标题
for i in ax.flat: #遍历取代之前的xticklabels的轮换。遍历每个子图的轴。
    plt.setp(i.get_xticklabels(), rotation=-90)
fig.tight_layout()

#处理和操作数据
#map方法；适用于序列数据，用来转变数据框的某个列（pandas序列）
df['class'] = df['class'].map({'Iris-setosa': 'SET',
                               'Iris-virginica': 'VIR',
                               'Iris-versicolor': 'VER'})
# map将一个字典作为参数，为每个单独的鸢尾花类型传入替代的文本。
# Apply方法；既可以在数据框上工作，也可以在序列上工作。
df['wide petal'] = df['petal width'].apply(lambda v: 1 if v >= 1.3 else 0) #序列
'''添加新的一列wide petal;如果花瓣宽度啊等于或宽于中位值，编码为1，否则编码为0'''
df['petal area'] = df.apply(lambda r: r['petal length'] * r['petal width'],axis=1)
'''数据调用apply; 参数axis=1对行进行操作，axis=0对列进行操作'''
# Applymap 对数据框里所有的数据单元执行一个函数。
df.applymap(lambda v: np.log(v) if isinstance(v, float) else v)
# 基于选择的类别对数据进行分组。
df.groupby('class').mean() #按照class对数据进行划分，并获得每个特征的均值。
df.groupby('class').describe() #每个类别完全的描述性统计信息。
df.groupby('petal width')['class'].unique().to_frame() #通过和每个唯一相关联的花瓣长度，对类别分组。
df.groupby('class')['petal width'].agg({'delta': lambda x: x.max() - x.min(),
                                        'max': np.max, 'min': np.min})
# 根据类别来分组花瓣宽度的时候，返回，max,min，及差值。
# 以字典的形式传递给 .agg()方法，以此返回一个将字典键值作为列名的数据框。

# 建模和评估
# Statsmodels 构建线性回归模型/花萼长度和宽度之间的模型。
# 散点图目测关系。
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(df['sepal width'][:50], df['sepal length'][:50])
ax.set_ylabel('Sepal Length')
ax.set_xlabel('Sepal Width')
ax.set_title('Setosa Sepal Width vs. Sepal Length', fontsize=14, y=1.02)
# 线性回归模型
import statsmodels.api as sm
y = df['sepal length'][:50]
x = df['sepal width'][:50]
X = sm.add_constant(x)

results = sm.OLS(y, X).fit()
print(results.summary())
# 绘制回归线
fig, ax = plt.subplots(figsize=(7,7))
ax.plot(x, results.fittedvalues, label='regression line')
ax.scatter(x,y, label='data point', color='r')
ax.set_ylabel('Sepal Length')
ax.set_xlabel('Sepal Width')
ax.set_title('Setosa Sepal Width vs. Sepal Length', fontsize=14, y=1.02)
ax.legend(loc=2)

# scikit-learn
from sklearn.ensemble import RandomForestClassifier #导入随机森林分类器
from sklearn.model_selection import train_test_split #将数据分成训练组和测试组的模块
clf = RandomForestClassifier(max_depth=5, n_estimators=10) #实例化分类器/10个决策树，每棵树最多5层判定深度。

X = df.ix[:, :4] #定义X矩阵
y = df.ix[:, 4] #定义y向量
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.3)
'''数据打乱并划分为4个子集，test_size=0.3意味着有数据集30%分配给x_test和y_test部分,其余分配到训练部分'''

clf.fit(x_train,y_train) #训练模型

y_pred = clf.predict(x_test) # 预测测试数据
# 创建对应实际标签和预估标签的数据框。
rf = pd.DataFrame(list(zip(y_pred, y_test)), columns=['predicted','actual'])
# 添加一列'correct',预测值与测试数据结果相同的为1，不相同的为0
rf['correct'] = rf.apply(lambda r: 1 if r['predicted'] == r['actual'] else 0, axis=1)
rf
# 正确的预测次数总和/测试样例的总数，即为正确率
rf['correct'].sum()/rf['correct'].count()
# 最佳辨别力
f_importances = clf.feature_importances_ #此方法返回特征在决策树中划分叶子节点的相对能力。
f_names = df.columns[:4]
f_std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
zz = zip(f_importances, f_names, f_std)
zzs = sorted(zz, key=lambda x: x[0], reverse=True)
imps = [x[0] for x in zzs]
labels = [x[1] for x in zzs]
errs = [x[2] for x in zzs]
plt.bar(range(len(f_importances)), imps, color='r', yerr=errs, align='center')
plt.xticks(range(len(f_importances)), labels)

# 支持向量机
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

clf = OneVsRestClassifier(SVC(kernel='linear'))
x = df.ix[:,:4]
y = np.array(df.ix[:,4]).astype(str) #SVM无法将标签解释为numpy的字符串

x_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3)


clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

rf = pd.DataFrame(list(zip(y_pred,y_test)), columns=['predicted', 'actual'])
rf['correct'] = rf.apply(lambda r: 1 if r['predicted'] == r['actual'] else 0, axis=1)
rf

# 构建应用程序，发现低价的公寓
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
pd.set_option('display.max_columns', 30) #设定指定选项的值
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.precision', 3)
CSV_PATH = r"D:/pythondemo/magic.csv"
df = pd.read_csv(CSV_PATH)
df.columns #展示列名
# 多单元公寓
mu = df[df['listingtype_value'].str.contains('Apartments For')] #提取df中包含Apartments For的矩阵
# 单单元公寓
su = df[df['listingtype_value'].str.contains('Apartment For')] #提取df中包含Apartment For的矩阵
len(mu)
len(su)
# 价格列无空值；解析卧室和浴室以及平方英尺/split字段
su['propertyinfo_value']
# 检查没有包含'bd'或'Studio'的行数
len(su[~(su['propertyinfo_value'].str.contains('Studio')\
         |su['propertyinfo_value'].str.contains('bd'))]) # ~按位取反运算符：对数据的每个二进制位取反,即把1变为0,把0变为1
'''
首先分别判断su中是否包含Studio或bd并获得两个布尔值列表
然后 | 或运算符计算两个列表，获得布尔值列表，这里全部为true
之后 ~ 运算符将所有true转换为false
su再选择为true的矩阵，意义为同时不包含bd和Studio的项；这里为空'''
# 检查没有包含'ba'的行数
len(su[~(su['propertyinfo_value'].str.contains('ba'))])
# 选择拥有浴室的房源
no_baths = su[~(su['propertyinfo_value'].str.contains('ba'))]
# 再排除缺失了浴室信息的房源
sucln = su[~su.index.isin(no_baths.index)]
# 使用项目符号进行切分
def parse_info(row):
    if not 'sqft' in row:
        br, ba = row.split('•')[:2] # split空格
        sqft = np.nan
    else:
        br, ba, sqft = row.split('•')[:3] # split " • " 这里是一个特殊字符，易出错
    return pd.Series({'Beds': br, 'Baths': ba, 'sqft': sqft})

attr = sucln['propertyinfo_value'].apply(parse_info) #在'propertyinfo_value'列上运行了apply函数
# 该操作返回一个数据框，其中每个公寓属性都会成为单独的列
# 在取值中将字符串删除
attr_cln = attr.applymap(lambda x: x.strip().split(' ')[0] if
isinstance (x, str) else np.nan)
attr
sujnd = sucln.join(attr_cln)
sujnd.T

### 面向对象
## 抽象




##
##PYTHON 数据分析实战
##

#函数式编程
map(function,list) #映射函数
filter(function,list) #过滤函数
reduce(function,list) #规约函数
'''lambda函数和列表生成式'''
items = [1,2,3,4,5]
list(map((lambda x: x + 1),items)) #对每一个元素进行操作
list(filter((lambda x: x < 4),items)) #只抽取函数返回结果为TRUE的列表元素
from functools import reduce
list(reduce((lambda x,y: x/y),items)) #对列表所有元素依次计算后返回唯一结果

## Numpy
# 多维数组
import numpy as np
a = np.array([1,2,3])
b = np.array([[1.3,2.4],[0.3,4.1]])
c = np.array(((1,2,3),(4,5,6)))

g = np.array([['a','b'],['c','d']])
g.dtype #dtype同时可以在array中接收参数。 定义数组。
np.zeros((3,3)) #shape参数指定维度信息、元素均为零的数组。
np.ones((3,3))
np.arange(0,10) #0-9
np.arange(4,10) #4-9
np.arange(0,12,3) #0-11 等差3
# 以上为一维数组
np.arange(0,12).reshape(3,4) #二维数组，根据参数拆成不同的部分。
np.linspace(0,10,5) #前两个参数指定序列的其实和结尾，第三个参数指定开头和结尾两个数字所指定的范围分成几个部分。
# 随机数填充数组
np.random.random(3) #单维
np.random.random(3,3) #多维

A = np.arange(0,9).reshape(3, 3)
B = np.ones((3,3))
A*B #元素级运算
np.dot(A,B) #矩阵积运算 或者 A.dot(B)

#通用函数 ufunc，对数组中的各个元素逐一进行操作；如np.sqrt() np.sin() np.log()
# 聚合函数，对一组数值进行操作，返回一个单一值作为结果的函数；如 np.sum()np.min()np.mean()np.std()
# 切片和索引抽取矩阵

# 数组迭代
for row in A:
    print(row)
for item in A.flat: #遍历矩阵的每个元素
    print(item)

# 通常用函数处理行，列或单个元素时，需要用到遍历。
# 如果用聚合函数处理每一行或列，返回一个数值作为结果。采用apply_along_axis()函数
np.apply_along_axis(np.mean, axis=0,arr=A) #对A数组的列求平均值，axis=0 列 axis=1 行
np.apply_along_axis(np.mean, axis=1,arr=A)

A = np.random.random((4,4))
A[A < .5] # 条件和布尔数组

# reshape()函数把一维数组转换为矩阵。
a = np.random.random(12)
A = a.reshape(3,4) #返回一个新数组
a.shape(3,4) #在原数组上修改
a = a.ravel() #改变数组可逆，修改回原一维数组
A.transpose() #矩阵转置

#数组操作
#多个数组整合在一起形成新数组。栈概念
# vstack() 执行垂直入栈操作，第二个数组作为行添加到第一个数组； hstack()执行水平入栈操作，作为列添加到第一个数组。
A = np.ones((3,3))
B = np.zeros((3,3))
np.vstack((A,B))
np.hstack((A,B))
# column_stack()和row_stack() 把一维数组作为列或行压入栈结构，形成新的二维数组。
a = np.array((0,1,2))
b = np.array((3,4,5))
c = np.array((6,7,8))
np.column_stack((a,b,c)) #按列
np.row_stack((a,b,c)) #按行
# hsplit()数组水平切分；vsplit()数组垂直切分
A = np.array(16).reshape(4,4)
[B,C] = np.hstack(A,2) #将数组A按照宽度2水平切分为两部分
[B,C] = np.vstack(A,2) #将数组A按照宽度2垂直切分为两部分
# split()可以把数组分为几个不对称的部分。
[A1,A2,A3] = np.split(A,[1,3],axis=1) #把矩阵A按axis=1列切分三部分，[1,3] 为分隔索引
[A1,A2,A3] = np.split(A,[1,3],axis=0) #按行

# save()方法以二进制格式保存数据，load()方法则从二进制文件中读取数据。
np.save('saved_data', data)
loaded_data = np.load('saved_data.npy')

# genfromtxt()从文本文件中读取数据并将其插入到数组中。
data = np.genfromtxt('data.csv', delimiter=',',names=True) #文件名/分隔符/列标题
data['id'] #列标题可以看成能够充当索引的标签。

##
##pandas库
##

import pandas as pd
import numpy as np

'''pandas核心为两大数据结构
Series         一维数据
DataFrame      多维数据
'''

# 调用Series()函数，以数组形式传入，就能创建一个Series对象。
s = pd.Series([12,-4,7,-9])
# 插入有意义的标签（默认从0开始的数值）
s = pd.Series([12,-4,7,-9], index=['a','b','c','d'])
# 分别查看
s.values
s.index
# 指定键或索引标签选择内部元素
s[2]
s['b']
s[0:2]
s[['b','c']]
# 赋值
s[1] = 0
# 定义新的Series对象；新对象是引用而不是副本
arr = np.array([1,2,3,4])
s3 = pd.Series(arr)
s4 = pd.Series(s)
# 筛选元素
s[s > 8]
# 运算
s/2
np.log(s) #数学函数必需指出出处np,并把Series实例作为参数传入。
# unique()函数去重
serd = pd.Series([1,0,2,1,2,3], index=['white','white','blue','green','green','yellow'])
serd.unique()
# value_counts()函数，返回各个不同的元素，并统计次数
serd.value_counts()
# isin()判断所属关系
serd.isin([0,3])
serd[serd.isin([0,3])]
# 缺失值 NaN
s2 = pd.Series([5,-3,np.NaN,14])
# isnull()和notnull() 识别没有对应元素的索引;取决于原Series对象的元素是否为NaN
# 这两个函数可用作筛选条件
s2.isnull()
s2[s2.isnull()]
s2[s2.notnull()]
# Series用作字典
mydict = {'red':2000,'blue':1000,'yellow':500,'orange':1000}
myseries = pd.Series(mydict)
# 可以单独指定索引；缺失值处添加NaN。
colors = ['red','yellow','orange','blue','green']
myseries = pd.Series(mydict,index=colors)

mydict2 = {'red':400,'yellow':1000,'black':700}
myseries2 = pd.Series(mydict2) #只求部分元素标签相同的两个series之和。
myseries + mydict2

# DataFrame对象
# 定义DataFrame对象
data = {'color':['blue','green','yellow','red','white'],
        'object':['ball','pen','pencil','paper','mug'],
        'price':[1.2,1.0,0.6,0.9,1.7]}
frame = pd.DataFrame(data)
# columns选项指定需要的列，选择自己感兴趣的数据。
frame2 = pd.DataFrame(data,columns=['object','price'])
# 标签放在数组中可以指定标签,index选项
frame3 = pd.DataFrame(data,index=['one','two','three','four','five'])
# 定义构造函数，指定三个参数：数据矩阵、index选项、columns选项
frame3 = pd.DataFrame(np.array(16).reshape(4,4),
                      index=['red','blue','yellow','white'],
                      columns=['ball','pen','pencil','paper'])
# 调用columns属性获得dataframe对象所有列的名称
frame.columns
frame.index
# values属性获取所有元素
frame.values
# 列名做索引选择指定列
frame['price'] # 返回series对象。
# 或以列名作为属性
frame.price
# ix属性和行的索引值获得
frame.ix[2] #返回series对象，列的名称变为索引数组的标签，列中的元素变为series的数据部分。
# 用一个数组指定多个索引值选择多行
frame.ix[[2,4]]
# 指定索引范围选择一部分
frame[0:3]
# 依次指定元素所在的列名称，行的索引值或标签。
frame['object'][3]
# index属性指定dataframe结构中的索引数组，columns属性指定包含列名称的行。
# name属性为这两个二级结构指定标签
frame.index.name = 'id'
frame.columns.name = 'item'
# 添加列；指定dataframe实例新列的名称，为其赋值。
frame['new'] = 12
# 借助np.array()函数预先定义一个序列，更新某一列的所有元素。
ser = pd.Series(np.array(5))
frame['new'] = ser
# 选择单个元素，为其赋新值
frame['price'][2] = 3.3
# 元素所属关系
frame.isin([1.0,'pen'])
# 返回包含布尔值的dataframe对象，将其作为条件，得到一个新的dataframe。
frame[frame.isin([1.0,'pen'])] # 只包含满足条件的元素，其余元素为NaN
# del命令删除一整列数据。
del frame['new']
# 指定条件筛选数据
frame[frame < 2]
# 嵌套字典生成对象。作为参数传递给DataFrame()构造函数，pandas将外部的键解释成列名称；
# 内部的键解释为用作索引的标签。
nestdict = {'red': {2012: 22, 2013: 33},
            'white': {2011: 13, 2012: 22, 2013: 16},
            'blue': {2011: 17, 2012: 27, 2013: 18}}
# 解释嵌套结构的时候，pandas会用NaN填补缺失的元素。
frame2 = pd.DataFrame(nestdict)
# Index对象声明后不可变
ser = pd.DataFrame([5,0,3,8,4], index=['red','blue','yellow','white','green'])
# idmin()和idmax()函数分别返回索引最小值和最大值。
serd = pd.DataFrame(range(6), index=['white','white','blue','green','green','yellow'])
# 从数据结构中选取元素时，如果一个标签对应多个元素，得到的将是一个series对象（dataframe对象）。
# is_unique属性检查是否有重复的索引项。
serd.index.is_unique
# index对象不可变，但是可以更换索引
ser = pd.Series([2,5,7,4], index=['one','two','three','four'])
ser.reindex(['three','four','five','one'])
ser3 = pd.Series([1,5,6,3],index=[0,3,5,6])
# 插入缺失索引，其元素为前面索引编号比它小的那一项元素。
ser3.reindex(range(6),method='ffill')
# 如果想用新插入索引后边的元素，使用bfill方法
ser3.reindex(range(6),method='bfill')
# 扩展到dataframe，更换行或列；用NaN弥补缺失的元素。
frame.reindex(range(5), columns=['colors','price','new','object'])
# drop()返回不包含已删除索引及其元素的新对象
ser = pd.Series(np.array(4), index=['red','blue','yellow','white'])
ser.drop('yellow')
# 删除多项
ser.drop(['blue','white'])
# 删除dataframe中的元素
frame = pd.DataFrame(np.arange(16).reshape((4,4)),
                     index=['red','blue','yellow','white'],
                     columns=['ball','pen','pencil','paper'])
frame.drop(['blue','yellow']) #传入行索引删除行
frame.drop(['pen','pencil'],axis=1) #需要axis=1指定删除列
# 算术和数据对齐
s1 = pd.Series([3,2,5,1],['white','yellow','green','blue'])
s2 = pd.Series([1,4,7,2,1],['white','yellow','black','blue','brown'])
s1 + s2
frame1 = pd.DataFrame(np.arange(16).reshape((4,4)),
                      index=['red','blue','yellow','white'],
                      columns=['ball','pen','pencil','paper'])
frame2 = pd.DataFrame(np.arange(12).reshape((4,3)),
                      index=['blue','green','white','yellow'],
                      columns=['mug','pen','ball'])
frame1 + frame2
# 算术运算方法；add(),sub(),div(),mul()
f = lambda x: x.max() - x.min()
def f(x):
    return x.max() - x.min()
frame.apply(f)
frame.apply(f,axis=1)
def f(x):
    return pd.Series([x.min(),x.max()],index=['min','max'])
frame.apply(f)
# 统计函数
frame.sum()
frame.mean()
frame.describe()
# 排序
ser = pd.Series([5,0,3,8,4], index=['red','blue','yellow','white','green'])
ser.sort_index()
ser.sort_index(ascending=False) #指定ascending选项，将其值置为False，则降序排列。
frame.sort_index(axis=1) # 按列进行排序
# 对数据结构中的元素进行排序。
# Series对象使用order()函数。
ser.order()
# DataFrame对象使用sort_index()函数，且需要by选项指定根据哪一列进行排序。
frame.sort_index(by='pen')
frame.sort_index(by=['pen','pencil']) #基于多列排序，以数组形式传入参数。
# 排位次操作
ser.rank()
ser.rank(method='first') #数据在数据结构中的顺序作为位次。
ser.rank(ascending=False) #降序
# 相关性和协方差
seq2 = pd.Series([3,4,3,4,5,4,3,2],['2006','2007','2008','2009','2010','2011','2012','2013'])
seq = pd.Series([1,2,3,4,4,3,2,1],['2006','2007','2008','2009','2010','2011','2012','2013'])
seq.corr(seq2) #相关性
seq.cov(seq2) #协方差

frame2 = pd.DataFrame([[1,4,3,6],[4,5,6,1],[3,3,1,5],[4,1,6,4]],
                      index=['red','blue','yellow','white'],
                      columns=['ball','pen','pencil','paper'])
frame2.corr() #单个DataFrame对象的相关性
frame2.cov() #单个DataFrame对象的协方差
# corrwith()方法可以计算DataFrame对象的列和行与Series对象或其他DataFrame对象元素两两之间的相关性。
frame2.corrwith(ser)
frame2.corrwith(frame)
# NaN数据
ser = pd.Series([0,1,2,np.NaN,9], index=['red','blue','yellow','white','green'])
ser['white'] = None
# dropna()函数删除所有的NaN
ser.dropna()
ser[ser.notnull()] #另一种方法用notnull()函数作为选取元素的条件，实现直接过滤。
# DataFrame使用dropna()方法，只要行或列有一个NaN元素，该行或列的全部元素都会被删除。
frame3 = pd.DataFrame([[6,np.nan,6],[np.nan,np.nan,np.nan],[2,np.nan,5]],
                      index = ['blue','green','red'],
                      columns = ['ball','mug','pen'])
frame3.dropna()
frame3.dropna(how='all') #how选项，指定其值为all，只删除所有元素均为NaN的行或列。

# 为NaN元素填充其他值
frame3.fillna(0) #将所有NaN替换为0
frame3.fillna({'ball':1,'mug':0,'pem':99}) #指定列名称及要替换成的元素

# 等级索引
# 创建包含两列索引的series对象。
mser = pd.Series(np.random.rand(8),
                 index=[['white','white','white','blue','blue','red','red','red'],
                        ['up','down','right','up','down','up','down','left']])
mser.index
mser['white'] #选取第一列索引中某一索引项的元素
mser[:,'up'] #选取第二列索引中某一索引项的元素
mser['white','up'] #选取某一特定的元素，指定两个索引

# unstack()函数调整dataframe中的数据，把使用的等级索引series对象转换为一个简单的dataframe对象。
# 第二列索引转换为相应的列。
mser.unstack()
# 逆操作，dataframe转换为series对象。
frame.stack()
# 声明DataFrame对象时，为index和columns选项分别制定一个元素为数组的数组（等级索引）。
mframe = pd.DataFrame(np.random.randn(16).reshape(4,4),
                      index=[['whire','white','red','red'],['up','down','up','down']],
                      columns=[['pen','pen','paper','paper'],[1,2,1,2]])
# swaplevel()函数以要呼呼阿奴位置的两个层级的名称为参数，返回交换位置后的一个新对象。
mframe.columns.names = ['objects','id'] #命名列名
mframe.index.names = ['colors','status'] #命名行名
mframe.swaplevel('colors','status') #交换位置
# sortlevel()只根据一个层级对数据排序。
mframe.sortlevel('colors')

# 按层级统计数据
mframe.sum(level='colors') #level选项指定要获取哪个层级的描述性和概括统计量。
mframe.sum(level='id', axis=1) #按列统计

# pandas数据读写
# I/O API工具
'''
        读取函数            写入函数
        read_csv            to_scv
        read_excel          to_excel
        read_hdf            to_hdf
        read_sql            to_sql
        read_json           to_json
        read_html           to_html
        read_stata          to_stata
        read_clipboard      to_clipboard
        read_pickle         to_pickle
        read_msgpack        to_msgpack
        read_gbq            to_gbq
'''
# 读取CSV或文本文件中的数据
csvframe = pd.read_csv('myCSV_01.csv') #读取内容，同时将其转换为DataFrame对象。
pd.read_table('ch05_01.csv',sep=',') #read_table()函数需指定分隔符。
# 没有表头的情况时使用header选项，将其值置为None，pandas会添加默认表头。
PATH = r'D:/pythondemo/PY_Analysis/'
pd.read_csv(PATH + 'ch05_02.csv', header=None)
# names选项指定表头，直接把存有各列名称的数组赋值给它即可。
pd.read_csv(PATH + 'ch05_02.csv',names=['white','red','blue','green','animal'])
# 等级索引
pd.read_csv(PATH+'CH05_03.csv', index_col=['color','status'])
# sep选项指定正则表达式，在read_table()函数内使用。
pd.read_table(PATH + 'ch05_05.txt', sep='\D*',header=None)
pd.read_table(PATH + 'ch05_06.txt', sep=',',skiprows=[0,1,3,6]) #排除指定行
# 读取部分数据
pd.read_csv(PATH + 'ch05_02.csv',skiprows=[2],nrows=3,header=None)
# 切分想要解析的文本，然后遍历各个部分，逐一对其执行某一特定操作。
out = pd.Series()
i = 0
pieces = pd.read_csv(PATH + 'ch05_01.csv',chunksize=3)
for piece in pieces:
    out.set_value(i,piece['white'].sum())
    i = i + 1
# 往CSV文件写入数据
frame2.to_csv('ch05_07.csv')
# 取消写入索引和列名称
frame2.to_csv('ch05_07.csv',index=False,header=False)
# 数据结构中NaN写入文件时为空字段，可以用na_rep选项把空字段替换为需要的值；常用值NULL、0、NaN
frame3.to_csv('ch05_09.csv',na_rep='NaN')

# 写入数据到HTML文件
frame = pd.DataFrame(np.arange(4).reshape(2,2))
print(frame.to_html())
# 在HTML文件中自动生成表格
frame = pd.DataFrame(np.random.random((4,4)),
                     index=['whitee','black','red','blue'],
                     columns=['up','down','right','left'])

s = ['<HTML>'] #创建包含HTML页面代码的字符串
s.append('<HEAD><TITLE> My DataFrame</TITLE><HEAD>')
s.append('<BODY>')
s.append(frame.to_html())
s.append('</BODY></HTML>')
html = ''.join(s)
# 写入到myFrame.html文件中
html_file = open('myFrame.html','w')
html_file.write(html)
html_file.close()

# 解析HTML文件
web_frames = pd.read_html('myFrame.html')
# read_html()函数最常用的模式是以网址作为参数，直接解析并抽取网页中的表格。

# 读取XML数据
'''
from lxml import objectify
xml = objectify.parse('books.xml')
'''
# 读写EXCEL文件
pd.read_excel(PATH + 'data.xls')
pd.read_excel('data.xls','Sheet2') #工作表名称
pd.read_excel('data.xls',1) #工作表序号

frame = pd.DataFrame(np.random.random((4,4)),
                     index=['exp1','exp2','exp3','exp4'],
                     columns=['Jan2015','Fab2015','Mar2015','Apr2015'])

frame.to_excel('data2.xlsx') #写入excel文件

# JSON数据
pd.read_json()
# HDF5数据
from pandas.io.pytables import HDFStore
store = HDFStore('mydata.h5')
# pickle-Python对象序列化
# 对接数据库

# 深入pandas 数据处理
# 执行合并操作的函数为merge()
frame1 = pd.DataFrame({'id':['ball','pencil','pen','mug','ashtray'],
                      'price':[12.33,11.44,33.21,13.23,33.62]})
frame2 = pd.DataFrame({'id':['pencil','pencil','ball','pen'],
                       'color':['white','red','red','black']})

pd.merge(frame1,frame2) #执行合并操作
pd.merge(frame1,frame2,on='id') #按指定列合并
pd.merge(frame1,frame2,left_on='id',right_on='sid') #分别指定对齐的列
pd.merge(frame1,frame2,on='id',how='outer') #连接类型用how选项指定。
pd.merge(frame1,frame2,on=['id','brand'],how='outer') #合并多个键

# 根据索引合并
pd.merge(frame1,frame2,right_index=True,left_index=True)
frame1.join(frame2) #列名不可有重合

# 拼接
array1 = np.arange(9).reshape((3,3))
array2 = np.arange(9).reshape((3,3))+6
np.concatenate([array1,array2],axis=1) #按列
np.concatenate([array1,array2],axis=0) #按行

ser1 = pd.Series(np.random.rand(4),index=[1,2,3,4])
ser2 = pd.Series(np.random.rand(4),index=[5,6,7,8])
pd.concat([ser1,ser2])
pd.concat([ser1,ser2],axis=1) #按列

pd.concat([ser1,ser3],axis=1,join='inner') #内连接,结构中无法识别重叠部分
pd.concat([ser1,ser2],axis=1,keys=[1,2]) #指定的键变为列名
pd.concat([ser1,ser2],keys=[1,2]) #指定的键创建等级序列

# frame方法相同
frame1 = pd.DataFrame(np.random.rand(9).reshape(3,3),index=[1,2,3],
                      columns=['A','B','C'])
frame2 = pd.DataFrame(np.random.rand(9).reshape(3,3),index=[4,5,6],
                      columns=['A','B','C'])

pd.concat([frame1,frame2])
pd.concat([frame1,frame2],axis=1)

# 组合
# combine_first()函数可以用来组合series对象，同时对齐数据。
ser1 = pd.Series(np.random.rand(5),index=[1,2,3,4,5])
ser2 = pd.Series(np.random.rand(4),index=[2,4,5,6])

ser.combine_first(ser2)
ser2.combine_first(ser1)
# 如果想进行部分合并，仅指定要合并的部分即可。
ser1[:3].combine_first(ser2[:3])

# 轴向旋转
# 入栈(stacking)：旋转数据结构，把列转换为行；出栈(unstacking)：把行转换为列。
frame1 = pd.DataFrame(np.arange(9).reshape(3,3),
                      index=['white','black','red'],
                      columns=['ball','pen','pencil'])

ser5 = frame1.stack() # 返回Series对象
ser5.unstack() # 重建DataFrame对象

# pivot()函数能够把长格式DataFrame转换为宽格式。
frame1 = pd.DataFrame(np.arange(9).reshape(3,3),
                      index=['white','black','red'],
                      columns=['ball','pen','pencil'])
# 删除一列；del命令
del frame1['ball']
# 删除多余的行；drop()函数
frame1.drop('white')

# 数据转换
# 删除重复元素
dframe = pd.DataFrame({'color':['white','white','red','red','white'],
                       'value':[2,1,3,3,2]})
# duplicated()函数检测重复的行，返回元素为布尔型的Series对象。
dframe[dframe.duplicated()]

# drop_duplicates()实现了删除功能；返回删除重复行后的DataFrame对象。
# tokens = [s.strip() for s in text.split(',')]

# 映射替换元素
frame = pd.DataFrame({'item':['ball','mug','pen','pencil','ashtray'],
                      'color':['white','rosso','verde','black','yellow'],
                      'price':[5.56,4.20,1.30,0.56,2.75]})
newcolors = {'rosso':'red','verde':'green'}
# replace()函数传入表示映射关系的字典作为参数。
frame.replace(newcolors)
# NaN替换为其他值。
ser = pd.Series([1,3,np.nan,4,6,np.nan,3])
ser.replace(np.nan,0)

# 用映射添加元素。
price ={'ball':5.56,'mug':4.20,'bottle':1.30,'scissors':3.41,'pen':1.30,
        'pencil':0.56,'ashtray':2.75}
frame['price'] = frame['item'].map(price)

# 重命名轴索引
# rename()函数，以表示映射关系的字典对象作为参数，替换轴的索引标签。
reindex = {0:'first',1:'second',2:'third','3':'fourth',4:'fifth'}
frame.rename(reindex)
# 若要重命名各列，必须使用columns选项。
recolumn = {'item':'object','price':'value'}
frame.rename(index=reindex,columns=recolumn)
frame.rename(index={1:'first'}, columns={'item':'object'}) #进一步指定赋值
frame.rename(index={1:'first'}, columns={'item':'object'}, inplace=True) #原位修改

# 离散化
bins = [0,25,50,75,100]
cat = pd.cut(results,bins)
# 统计每个面元的出现次数，即每个类别有多少元素。
pd.value_counts(cat)
bin_names = ['unlikely','less_likely','likely','highly likely']
pd.cut(results,bins,labels=bin_names) #指定面元的名称
# qcut()函数能够保证每个面元的个体数相同，但每个面元的区间大小不等。
quintiles = pd.qcut(results,5)

# 异常值检测和过滤
randframe = pd.DataFrame(np.random.randn(1000,3))
# describe()函数查看每一列的描述性统计量。
randframe.describe()
# std()函数求得每一列的标准差。
randframe.std()
randframe[(np.abs(randframe) > (3*randframe.std())).any(1)]

# 排序
# numpy.random.permutation()函数调整Series对象或DataFrame对象各行的顺序
nframe = pd.DataFrame(np.arange(25).reshape(5,5))
new_order = np.random.permutation(5) #创建一个顺序随机的5个整数的数组。
nframe.take(new_order)

# 随机取样
sample = np.random.randint(0,len(nframe),size=3)
nframe.take(sample)

##
## 字符串处理
# split()函数以参考点为分隔符。
text = '16 Bolton Avenue , Boston'
text.split(',')
# strip()函数删除多余的空白字符。
tokens = [s.strip() for s in text.split(',')]

# 多个字符串拼接；join()函数。
strings = ['A+','A-','B','BB','BBB','C+']
str.join(strings)
# 查找字符串；in关键字。
'Boston' in text
# 字符串查找函数
text.index('Boston') #找不到字符串报错
text.find('Boston') #找不到字符串返回-1
text.count('e') #统计字符串组合在文本中的出现次数
text.replace('Avenue','Street') #替换或删除字符串中的子串。

# 正则表达式
import re
# 一个或多个空白字符；\s+
text = "This is    an\t odd \n text!"
re.split('\s+', text)
# 预先编译正则表达式，提升效率。
regex = re.compile('\s+')
regex.split(text)

text = 'This is my address: 16 Bolton Avenue, Boston'
re.findall('A\w+',text) #找出字符串中所有以大写字母A开头的单词。
re.findall('[A,a]\w+',text)

# fandall()函数返回一列所有符合模式的子串，search()函数仅返回第一处符合模式的子串。
re.search('[A,a]\w+',text) #可迭代对象
search = re.search('[A,a]\w+',text)
search.start()
search.end()
text[search.start():search.end()]
# match()函数从字符串开头开始匹配；如果第一个字符就不匹配，它不会再搜索字符串内部。
re.match('[A,a]\w+',text)

# 数据聚合
# GroupBy
# groupby()函数实现：分组、函数处理、合并
frame = pd.DataFrame({'color':['white','red','green','red','green'],
                      'object':['pen','pencil','pencil','ashtray','pen'],
                      'price1':[5.56,4.20,1.30,0.56,2.75],
                      'price2':[4.75,4.12,1.60,0.75,3.15]})
# 使用color列的组标签，计算price1列的均值。
group = frame['price1'].groupby(frame['color']) #先获取price1列，然后指定color列。
group.mean() #对每组进行操作
group.sum()

# 等级分组
ggroup = frame['price1'].groupby([frame['color'],frame['object']])
ggroup.sum()
# 一次指定所有需要的分组依据和计算方法。
frame[['price1','price2']].groupby(frame['color']).mean()

# 支持迭代，生成一系列由各组名称及其数据部分组成的元组。
for name,group in frame.groupby('color'):
    print (name,group)

# 执行聚合操作后，可以再列名称前加上描述操作类型的前缀。
means = frame.groupby('color').mean().add_prefix('mean_')
group = frame.groupby('color')
group['price1'].quantile(0.6)

def range(series):
    return series.max() - series.min()
group.agg(range)
group.agg(['mean','std',range]) #分别为DataFrame对象添加相应的新列。

# 高级数据聚合  transform() apply()
frame = pd.DataFrame({'color':['white','red','green','red','green'],
                      'price1':[5.56,4.20,1.30,0.56,2.75],
                      'price2':[4.75,4.12,1.60,0.75,3.15]})
sums = frame.groupby('color').sum().add_prefix('tot_')
pd.merge(frame,sums,left_on='color',right_index=True)
# 替代方法，transform()方法可以根据dataframe对象每一行的关键字显示聚合结果。
frame.groupby('color').transform(np.sum).add_prefix('tot_')

frame = pd.DataFrame({'color':['white','black','white','white','black','black'],
                      'status':['up','up','down','down','down','up'],
                      'value1':[12.33,14.55,22.34,27.84,23.40,18.33],
                      'value2':[11.23,31.80,29.99,31.18,18.25,22.44]})
frame.groupby(['color','status']).apply(lambda x: x.max())
frame.rename(index=reindex, columns=recolumn)
temp = pd.date_range('1/1/2015',periods=10, freq='H')
timeseries = pd.Series(np.random.rand(10), index=temp)
timetable = pd.DataFrame({'date':temp, 'value1':np.random.rand(10),
                          'value2':np.random.rand(10)})


# 数据可视化
#  matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.plot([1,2,3,4])
plt.plot([1,2,3,4],[1,4,9,16],'ro')
plt.axis([0,5,0,20]) #x,y轴范围；【xmin, xmax, ymin, ymax】
plt.title('My first plot') #标题

import math
t = np.arange(0,2.5,0.1)
y1 = map(math.sin,math.pi*t)
y2 = map(math.sin,math.pi*t+math.pi/2)
y3 = map(math.sin,math.pi*t-math.pi/2)
plt.plot(t,list(y1),'b*',t,list(y2),'g*',t,list(y3),'y*')
plt.plot(t,list(y1),'b--',t,list(y2),'g',t,list(y3),'y-.')

# 使用关键字参数
# 改变线条粗细；linewidth=2.0

# 处理多个figure和Axes对象
# subplot()函数用参数设置分区模式和当前子图。
plt.subplots(211) #2*1子图的上图
plt.subplots(212) #2*1子图的下图

# 添加文本
plt.xlabel('Counting') #x轴标签
plt.ylabel('Square values') #y轴标签

plt.title('My first plot',fontsize=20,fontname='Times New Roman') #修改字体，字号
plt.xlabel('Counting',color='gray') #修改x轴标签颜色
# text(x,y,s,fontdict=None,**kwargs) 前两个参数为坐标，s为要添加的字符串。fontdict字体
plt.text(1,1.5,'First') #添加文本
plt.text(2,4.5,'Second')
plt.text(3,9.5,'Third')
plt.text(4,16.5,'Fourth')
# 添加公式；置于两个$符号之间自动识别
plt.text(1.1,12,r'$y=x^2$',fontsize=20,bbox={'facecolor':'yellow','alpha':0.2})
# 添加网格；grid()函数
plt.grid(True)
# 添加图例；legend()函数
plt.legend(['first series']) #默认右上角。位置由loc关键字控制。
'''
0   最佳位置
1   右上角
2   左上角
3   右下角
4   左下角
5   右侧
6   左侧垂直居中
7   右侧垂直居中
8   下方水平居中
9   上方水平居中
10  正中间
'''
# 多个序列，legend()顺序应与序列顺序保持一致
plt.legend(['First series','Second series','Third series'],loc=2)

# 保存图表
# 可以保存代码、网页；或者直接保存为图片：savefig()
plt.savefig('my_chart.png')

# 处理日期值；如月份
'''
ax.xaxis.set_major_locator(months)
ax.xaxis.set_majir_formatter(timeFmt)
ax.xaxis.set_major_locator(days)
'''

# 线型图
plt.plot(x,y,'k--',color='#87a3cc',linewidth=3)
# 笛卡尔坐标轴；gca()
ax=plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

# annotata()函数添加注释。
# LaTeX表达式，要在图形中显示的字符串；注释在图表中的位置用[x,y]传给xy关键字；
# 文本注释跟所解释的数据点的距离用xytext关键字参数指定。

plt.annotate(r'$\lim_{x\to 0}\frac{\sin(x)}{x}= 1$', xy=[0,1],xycoords='data',
             xytext=[30,30],fontsize=16,textcoords='offset point',arrowprops=dict(arrowstyles="->",
             connectionstyle="arc3,rad=.2"))

# 线型图；plot()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
data = {'series1':[1,3,4,3,5],
        'series2':[2,4,5,2,4],
        'series3':[3,2,3,1,3]}
df = pd.DataFrame(data)
x = np.arange(5)
plt.axis([0,5,0,7])
plt.plot(x,df)
plt.legend(data,loc=2)

# 直方图；hist()

pop = np.random.randint(0,100,100)
n,bins,patches = plt.hist(pop,bins=20)

# 条状图；bar()
index = [0,1,2,3,4]
values = [5,7,3,4,6]
plt.bar(index,values)
# xticks()刻度标签标明其类别。
plt.xticks(index+0.4,['A','B','C','D','E'])
# 关键字参数实现改进
std1 = [0.8,1,0.4,0.9,1.3]
plt.title('A bar Chart') #标题
plt.bar(index,values,yerr=std1,error_kw={'ecolor':'0.1',
                                         'capsize':6},alpha=0.7,label='First')
# yerr：标准差；erroe_kw：误差线；eColor：误差线颜色；capsize：误差线两头横线的宽度；alpha：透明度
plt.xticks(index+0.4,['A','B','C','D','E'])
plt.legend(loc=2)

# 水平条状图；barh()函数，水平条状图中，类别分布在y轴上，数值显示在x轴上
plt.title('A Horizontal Bar Chart')
plt.barh(index,values,serr=std1,error_kw={'ecolor':'0.1','capsize':6},alpha=0.7,
         label='First')
plt.yticks(index+0.4,['A','B','C','D','E'])
plt.legend(loc=5)

# 多序列条状图
index= np.arange(5)
values1 = [5,7,3,4,6]
values2 = [6,6,4,5,7]
values3 = [5,6,5,4,6]
bw = 0.3
plt.axis=([0,5,0,8])
plt.title('A Multiseries Bar Chart',fontsize=20)
plt.bar(index,values1,bw,color='b')
plt.bar(index+bw,values2,bw,color='g')
plt.bar(index+2*bw,values3,bw,color='r')
plt.xticks(index+1.5*bw,['A','B','C','D','E'])
# 水平图同单条状图

# 多序列堆积条状图；bar()函数中添加bottom关键字参数，把每个序列赋给相应的bottom关键字参数。
series1 = np.array([3,4,5,3])
series2 = np.array([1,2,2,5])
series3 = np.array([2,3,3,4])
index = np.arange(4)
plt.axis([0,4,0,15])
plt.bar(index,series1,color='r')
plt.bar(index,series2,color='b',bottom=series1)
plt.bar(index,series3,color='g',bottom=(series2+series1))
plt.xticks(index+0.4,['Jan15','Feb15','Mar15','Apr15'])
# 水平堆积条状图，同样使用barh()函数替换bar()函数。
# 不同的影线填充条状图，首先条状图颜色设置为白色，hatch关键字参数指定影线的类型。
# 影线字符（|， /， -， |， *）；同一符号出现的次数越多，则线条越密集。
plt.barh(index,series1,color='w',hatch='xx')
plt.barh(index,series2,color='w',hatch='///',left=series1) #水平堆积条状图
plt.barh(index,series3,color='w',hatch='\\\\\\',left=(series1+series2))
plt.yticks(index+0.4,['Jan15','Feb15','Mar15','Apr15'])
# pandas直接使用df.plot(kind='car,stacked=True)绘制堆积条状图

# 对比条状图，以x轴或y轴对称（其中一个序列y值取相反数）
import matplotlib.pyplot as plt
x0 = np.arange(8)
y1 = np.array([1,3,4,6,4,3,2,1])
y2 = np.array([1,2,5,4,3,3,2,1])
plt.ylim(-7,7)
plt.bar(x0,y1,0.9,facecolor='r',edgecolor='w') # 边框和内部颜色
plt.bar(x0,-y2,0.9,facecolor='b',edgecolor='w')
plt.xticks(())
plt.grid(True)
for x, y in zip(x0, y1):
    plt.text(x + 0.4, y + 0.05, '%d' %y, ha='center', va='bottom') #ha和va关键字调整标签的位置
for x, y in zip(x0, y2):
    plt.text(x + 0.4, y + 0.05, '%d' %y, ha='center', va = 'top')
plt.show()

# 饼图；pie()
import matplotlib.pyplot as plt
labels = ['Nokia','Samsung','Apple','Lumia']
values = [10,30,45,15]
colors = ['yellow','green','red','blue']
plt.pie(values,labels=labels,colors=colors)
plt.axis('equal') #设定为正圆
# 制作从饼图中抽取出一块的效果
explode = [0.3,0,0,0] #0~1范围，0表示没有抽取 1 为完全脱离
plt.pie(values,labels=labels,colors=colors,explode=explode,startangle=180) #startangle表示转换的角度（0-360）
plt.axis('equal')
# autopct关键字参数在每一块的中间位置添加文本标签来显示百分比。
plt.pie(values,labels=labels,colors=colors,explode=explode,
        shadow=True,autopct='%1.1f%%',startangle=180) #shadow添加阴影效果。
# DataFrame绘制饼图
df['series1'].plot(kind='pie',figsize=(6,6))

# 高级图表
# 等值线图（等高线图）；z = f(x,y)生成三维结构；contour()函数生成三维结构表面的等值线图。
import matplotlib.pyplot as plt
import numpy as np
dx = 0.01; dy = 0.01
x = np.arange(-2.0,2.0,dx)
y = np.arange(-2.0,2.0,dy)
X,Y = np.meshgrid(x,y)
def f(x,y):
    return (1 - y**5 + x**5)*np.exp(-x**2-y**2)
C = plt.contour(X,Y,f(X,Y),8,colors='black')
plt.contourf(X,Y,f(X,Y),8,cmap=plt.cm.hot) #cmap选定颜色
plt.clabel(C,inline=1,fontsize=10)
plt.colorbar() #颜色说明

# 极区图；bar()函数，传入角度θ列表和半径列表。
N = 8
theta = np.arange(0.,2*np.pi,2*np.pi/N)
radii = np.array([4,7,5,3,1,5,6,7])
plt.axes([0.025,0.025,0.95,0.95], polar=True)
colors = np.array(['lightgreen','#darkred','navy','brown','violet','plum',
                   'yellow','darkgreen'])
bars = plt.bar(theta, radii, width=(2*np.pi/N), bottom=0.0,color=colors)

# 3D数据可视化
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-2,2,0.1)
Y = np.arange(-2,2,0.1)
X,Y = np.meshgrid(X,Y)
def f(x,y):
    return (1 - y**5 + x**5)*np.exp(-x**2-y**2)
ax.plot_surface(X,Y,f(X,Y),rstride=1,cstride=1)
# cmap 关键字指定颜色，view——init()函数旋转曲面
# elev和azim两个关键字参数，从不同的视角查看曲面
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-2,2,0.1)
Y = np.arange(-2,2,0.1)
X,Y = np.meshgrid(X,Y)
def f(x,y):
    return (1 - y**5 + x**5)*np.exp(-x**2-y**2)
ax.plot_surface(X,Y,f(X,Y),rstride=1,cstride=1,cmap=plt.cm.hot)
ax.view_init(elev=30,azim=125)
# 3D散点图；scatter()函数
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
xs = np.random.randint(30,40,100)
ys = np.random.randint(20,30,100)
zs = np.random.randint(10,20,100)
xs2 = np.random.randint(50,60,100)
ys2 = np.random.randint(30,40,100)
zs2 = np.random.randint(50,70,100)
xs3 = np.random.randint(10,30,100)
ys3 = np.random.randint(40,50,100)
zs3 = np.random.randint(40,50,100)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(xs,ys,zs)
ax.scatter(xs2,ys2,zs2,c='r',marker='^')
ax.scatter(xs3,ys3,zs3,c='g',marker='*')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 3D条状图；bar()函数
x = np.arange(8)
y = np.random.randint(0,10,8)
y2 = y + np.random.randint(0,3,8)
y3 = y2 + np.random.randint(0,3,8)
y4 = y3 + np.random.randint(0,3,8)
y5 = y4 + np.random.randint(0,3,8)
clr = ['#4bb2c5','#c5b47f','#EAA228','#579575','#839557','#958C12','#953579','#4b5de4']
fig = plt.figure()
ax = Axes3D(fig)
ax.bar(x,y,0,zdir='y',color=clr)
ax.bar(x,y2,10,zdir='y',color=clr)
ax.bar(x,y3,20,zdir='y',color=clr)
ax.bar(x,y4,30,zdir='y',color=clr)
ax.bar(x,y5,40,zdir='y',color=clr)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.view_init(elev=40)

# 多面板图形
# 在其他子图中显示子图；主Axes对象，跟放置另一个Axes对象实例的框架分开。
# 用figure()函数取到figure对象，用add_axes()函数在它上面定义两个Axes对象。
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])
inner_ax = fig.add_axes([0.6,0.6,0.25,0.25])
x1 = np.arange(10)
y1 = np.array([1,2,7,1,5,2,4,2,3,1])
x2 = np.arange(10)
y2 = np.array([1,3,4,5,4,5,2,6,4,3])
ax.plot(x1,y1,color='r')
inner_ax.plot(x2,y2,color='b')
# 子图网格；GridSpec()函数可以y用来管理更为复杂的情况，
# 可以把绘图区域分层多个子区域，把一个或多个子区域分配给每一幅子图。
gs = plt.GridSpec(3,3)
fig = plt.figure(figsize=(6,6))
fig.add_subplot(gs[1,:2])
fig.add_subplot(gs[0,:2])
fig.add_subplot(gs[2,0])
fig.add_subplot(gs[:2,2])
fig.add_subplot(gs[2,1:])
# 分配给子图，在add_subplot()函数返回的Axes对象上调用plot()函数。
gs = plt.GridSpec(3,3)
fig = plt.figure(figsize=(6,6))
x1 = np.array([1,3,2,5])
y1 = np.array([4,3,7,2])
x2 = np.arange(5)
y2 = np.array([3,2,4,6,4])
s1 = fig.add_subplot(gs[1,:2])
s1.plot(x,y,'r')
s2 = fig.add_subplot(gs[0,:2])
s2.bar(x2,y2)
s3 = fig.add_subplot(gs[2,0])
s3.barh(x2,y2,color='g')
s4 = fig.add_subplot(gs[:2,2])
s4.plot(x2,y2,'k')
s5 = fig.add_subplot(gs[2,1:])
s5.plot(x1,y1,'b^',x2,y2,'yo')

# 用scikit-learn库实现机器学习
'''
有监督学习；训练集包含作为预测结果（目标值）的额外的属性信息。这些信息可以指导模型对新数据（测试集）
作出跟已有数据类似的预测结果。
场景：分类；回归
无监督学习：训练集数据由一系列输入值x组成，其目标值未知。
场景：聚类；降维
'''
# 训练估计器用fit(x,y)函数，预测由predict(x)完成。
from sklearn import datasets
iris = datasets.load_iris()

x = iris.data[:,0]
y = iris.data[:,1]
species = iris.target
x_min,x_max = x.min() - .5,x.max() + .5
y_min,y_max = y.min() - .5,y.max() + .5
plt.figure()
plt.title('Iris Dataset - Classfication By Petal Sizes', size=14)
plt.scatter(x,y,c=species)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.xticks(())
plt.yticks(())

# 主成分分析法；PCA
# fit_transform()函数降维，属于PCA对象。
# 导入skearn.decomposiion模块，使用PCA()构造函数，n_components指定降到几维。
# 调用fit_transform()函数，传入数据作为参数。

from sklearn.decomposition import PCA
x_redeced = PCA(n_components=3).fit_transform(iris.data)
from mpl_toolkits.mplot3d import Axes3D
iris = datasets.load_iris()
x = iris.data[:,1]
y = iris.data[:,2]
species = iris.target
x_redeced = PCA(n_components=3).fit_transform(iris.data)
fig = plt.figure()
ax = Axes3D(fig)
ax.set_title('Iris Dataset by PCA',size=14)
ax.scatter(x_redeced[:,0],x_redeced[:,1],x_redeced[:,2],c = species)
ax.set_xlabel('First eigenvector')
ax.set_ylabel('Second eigenvector')
ax.set_zlabel('Third eigenvector')
ax.w_xaxis.set_ticklabels(()) # 设置轴标签值
ax.w_yaxis.set_ticklabels(())
ax.w_zaxis.set_ticklabels(())

# k近邻算法预测
np.random.seed(0)
iris = datasets.load_iris()
x = iris.data
y = iris.target
i = np.random.permutation(len(iris.data)) #随机打乱数据
x_train = x[i[:-10]] #选择训练数据
y_train = y[i[:-10]]
x_test = x[i[-10:]] #选择测试数据
y_test = y[i[-10:]]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',
                     metric_params=None,n_neighbors=5,p=2,weights='uniform')
knn.predict(x_test) #预测

# Diabetes 数据集
from sklearn import datasets
diabetes = datasets.load_diabetes()
diabetes.data[0]
'''
这些数据是经过特殊处理得到的，10个数据中的每一个都做了均值中心化处理，然后又用标准差乘以个体数量
调整了数值范围。任何一列的所有数值之和为1.
np.sum(diabetes.data[:,0]**2)
规范化处理不会失去价值或丢失统计信息。
'''
# 线性回归：最小平方回归
'''
线性回归指的是用训练集数据创建线性模型的过程，最简单的形似则是基于（y = a*x + c）刻画的直线
方程。在计算参数a和c时，以最小化残差平方和为前提。
'''
# 先导入linear_model模块，然后用LinearRegression()构造函数创建预测模型。
from sklearn import linear_model
linreg = linear_model.LinearRegression()
# 测试
diabetes = datasets.load_diabetes()
x_train = diabetes.dara[:-20]
y_train = diabetes.target[:-20]
x_test = diabetes.data[-20:]
y_test = diabetes.data[-20:]
# 在预测模型上调用fit()函数，使用训练集做训练。
linreg.fit(x_train,y_train)
# 训练结束后，调用预测模型的coef_属性，可以得到回归系数b.
linreg.coef_
# predict()函数传入测试集作为参数，得到预测目标值。
linreg.predict(x_test,y_test)
# 方差是评价预测结果好坏的一个不错的指标。方差越接近于1，说明预测结果越准确。
linreg.score(x_test,y_test)

# 对每个生理特征进行回归分析，创建10个模型
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import datasets
linreg = linear_model.LinearRegression()
diabetes = datasets.load_diabetes()
x_train = diabetes.data[:-20]
y_train = diabetes.target[:-20]
x_test = diabetes.data[-20:]
y_test = diabetes.data[-20:]
plt.figure(figsize=(8,12))
for f in range(0,10):
    xi_test = x_test[:,f]
    xi_train = x_train[:,f]
    xi_test = xi_test[:,np.newaxis]
    xi_train = xi_train[:,np.newaxis]
    linreg.fit(xi_train,y_train)
    y = linreg.predict(xi_test)
    plt.subplots(5,2,f+1)
    plt.scatter(xi_test,y_test,color='k')
    plt.plot(xi_test,color='b',linewith=3)

# 支持向量机；二元判别模型
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
x = np.array([[1,3],[1,2],[1,1.5],[1.5,2],[2,3],[2.5,1.5],
             [2,1],[3,1],[3,2],[3.5,1],[3.5,3]])
y = [0]*6 + [1]*5
svc = svm.SVC(kernel='linear').fit(x,y)
X,Y = np.mgrid[0:4:200j,0:4:200j]
# decision_function()函数绘制决策边界。
Z = svc.decision_function(np.c_[X.ravel(),Y.ravel()])
Z = Z.reshape(X.shape)
plt.contourf(X,Y,Z > 0,alpha=0.4)
plt.contour(X,Y,Z,colors=['k'],linestyles=['-'],levels=[0])
plt.scatter(x[:,0],x[:,1],c=y,s=50,alpha=0.9)
'''
正则化是一个与SVC算法相关的概念，用参数C来设置：C值较小，表示计算间隔时，将分界线两侧的大量
甚至全部数据点都考虑在内（泛化能力强）；C值较大，表示只考虑分界线附近的数据点（泛化能力弱）。
若不指定C值，默认它的值为1。
'''
# 非线性SVC；内核 poly;另一种内核 rbf

# SVC方法经扩展可用来解决回归问题，这种方法称为支持向量回归SVR。
# 测试数据集数据必须按升序形式排列。












