# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:04:36 2017
基于完整的乳腺癌数据，利用pandas进行数据预处理，logstic回归的两种方式、加正则化项、感知机算法
索引方法：
reindex(index=None,**kwargs) **kwargs：method=None,fill_value=np.NaN
method:{'backfill', 'bfill', 'pad', 'ffill', None} 参数用于指定插值（填充）方式，当没有给出时，
自动用 fill_value 填充，默认为 NaN（ffill = pad，bfill = back fill，分别指插值时向前还是向后取值）
行索引：
ser = Series([4.5,7.2,-5.3,3.6],index=['d','b','a','c'])
 a = ['a','b','c','d','e']
 ser.reindex(a)
 
pandas中的数据结构DataFrame
df=pd.DataFrame(np.random.randn(3,4)) 创建方式
# 获得行索引信息
df.index
# 获得列索引信息
df.columns
# 获得df的size
df.shape
# 获得df的行数
df.shape[0]
# 获得df的列数
df.shape[1]
# 获得df中的值
df.values

列索引：
data={'a':[1,3,5,7],'b':[2,4,6,8]}
df = DataFrame(data)
@author: mjw
处理缺失数据，pandas中NA表现为np.nan,python内建的None也会被当做NA处理
处理NA的方式：dropna,fillna,isnull,notnull
is (not) null
dropna
dropna(axis=0, how='any', thresh=None),all切片元素全为NA时抛弃改行，thresh = 3,一行中至少有3个非NA值时将其保留
fillna(value=None, method=None, axis=0)

data0.ix[:, (data0 != data0.ix[0]).any()] 删除一列全相同的数据
"""
import pandas as pd
import numpy as np

column_names  = ['Sample code number','Clump Thickness','Uniformity of Cell Size',
                 'Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size',
                 'Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names = column_names)
#将？替换为标准缺失值表示
data = data.replace(to_replace = '?',value = np.nan)
#丢弃带有缺失值的数据（只要有一个维度有缺失）
data = data.dropna(how = 'any')
#print(data.shape)
from sklearn.cross_validation import train_test_split
#随机采样25%的数据用于测试，剩下75%用作训练
#train_test_split是交叉验证中常用的函数，功能是从样本中随机的按比例选取train data和testdata
X_train,X_test,y_train,y_test = train_test_split(data[column_names[1:10]],data[column_names[10]],test_size = 0.25,random_state = 33)
#y_train.value_counts()   y_test.value_counts()

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import  Perceptron

ss = StandardScaler()      #零均值归一化
X_train = ss.fit_transform(X_train )  #fit是为后续的API函数服务的
X_test = ss.transform(X_test)
lr = LogisticRegression()
lr1 = LogisticRegression(C = 1,random_state=0)  #默认l2正则化项
lr2 = LogisticRegression(penalty = 'l1',C = 1,random_state=0)  #l1正则化（稀疏化）
sgdc = SGDClassifier()
ppn = Perceptron(n_iter = 40, eta0 = 0.1, random_state = 0) #感知机参数，迭代次数、学习率、random_data在每次迭代后重排训练数据集
lr.fit(X_train,y_train)
lr_y_predict = lr.predict(X_test)#lr.score(testx,testy)
lr1.fit(X_train,y_train)
lr1_y_predict = lr.predict(X_test)
lr2.fit(X_train,y_train)
lr2_y_predict = lr.predict(X_test)
sgdc.fit(X_train,y_train)
sgdc_y_predict = sgdc.predict(X_test)
ppn.fit(X_train,y_train)
perc_y_predict = ppn.predict(X_test)

from sklearn.metrics import classification_report
#
#在sklearn的LogisticRegression里，是通过每次增加负样本权重【】的方式，
#让样本“重新排列”，之后得到了一个新的theta，直到两次样本基本上都不怎么变
print ('Accuracy of LR Classifier:',lr.score(X_test,y_test))
print (classification_report(y_test,lr_y_predict,target_names = ['Benign','Malignant']))
print ('Accuracy of LR Classifier with L2 regularization:',lr1.score(X_test,y_test))
print (classification_report(y_test,lr1_y_predict,target_names = ['Benign','Malignant']))
print ('Accuracy of LR Classifier with L1 regularization:',lr2.score(X_test,y_test))
print (classification_report(y_test,lr2_y_predict,target_names = ['Benign','Malignant']))
print (lr2.coef_)
print ('Accuracy of SGD Classifier:',sgdc.score(X_test,y_test))
print (classification_report(y_test,sgdc_y_predict,target_names = ['Benign','Malignant']))
print ('Accuracy of Perceptron:',ppn.score(X_test,y_test))
print (classification_report(y_test,perc_y_predict,target_names = ['Benign','Malignant']))
print('Misclassified samples in Perceptron: %d' %(y_test != perc_y_predict).sum())

"""
加入正则化项(L2正则化)
"""
#权重系数随惩罚系数的变化情况（1/C）
import matplotlib.pyplot as plt
weights,params = [],[]
for c in np.arange(-5,5):
    lr = LogisticRegression(C=10**c,random_state=0) #C为正则化系数的倒数
    lr.fit(X_train,y_train)
    weights.append(lr.coef_[0])
    params.append(10**c)
weights = np.array(weights)
plt.plot(params,weights[:,0],label = 'petal length')
plt.plot(params,weights[:,1],linestyle = '--',label = 'petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc = 'upper left')
plt.xscale('log')
plt.show()
