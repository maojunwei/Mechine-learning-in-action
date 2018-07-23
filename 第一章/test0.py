# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:52:58 2017

@author: mjw
良/恶性乳腺癌肿瘤预测
Pands:针对于数据处理和分析的python工具包（数据读写、清洗、填充以及分析）
loc-通过标签索引行数据
eg. import pandas as pd  
    data = [[1,2,3],[4,5,6]]  
    index = ['d','e']  
    columns=['a','b','c']  
    df = pd.DataFrame(data=data, index=index, columns=columns)
    a = df.loc['d'][['b','c']]
"""
import pandas as pd
#调用pandas工具包的read_csv函数、模块
df_train = pd.read_csv('E:\Datasets\Breast-Cancer\Breast-Cancer-train.csv')
df_test = pd.read_csv('E:\Datasets\Breast-Cancer\Breast-Cancer-test.csv')
#选取‘Clump Thickness’与‘Cell Size’作为特征，构建正负分类样本
df_test_negaive = df_test.loc[df_test['Type'] == 0][['Clump Thickness','Cell Size']]
df_test_positive = df_test.loc[df_test['Type'] == 1][['Clump Thickness','Cell Size']]

import matplotlib.pyplot as plt
#绘制良性样本点
#柱状图bar,散点图scatter,s散点的大小
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker = 'o',s = 20, c = 'red')
#绘制恶性样本点
plt.scatter(df_test_negaive['Clump Thickness'],df_test_negaive['Cell Size'],marker = 'x',s = 20, c = 'black')
#绘制x,y轴的说明
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()

import numpy as np
#产生随机数
intercept = np.random.random([1])
coef = np.random.random([2])
lx = range(0,12)
ly = (-intercept - lx * coef[0]) / coef[1]
plt.plot(lx,ly,c = 'yellow')

plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker = 'o',s = 20, c = 'red')
plt.scatter(df_test_negaive['Clump Thickness'],df_test_negaive['Cell Size'],marker = 'x',s = 20, c = 'black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(df_train[['Clump Thickness','Cell Size']][:10],df_train['Type'][:10])
print ('Testing accuracy (10 training samples):', lr.score(df_test[['Clump Thickness','Cell Size']],df_test['Type']))

#分类面为 lx*coef[0] + ly * coef[1] + intercept = 0,映射到2维平面之后为：
# ly = (-intercept - lx * coef[0])/coef[1]
intercept = lr.intercept_
coef = lr.coef_[0,:]  #多维数组
ly = (-intercept - lx * coef[0])/coef[1]

plt.plot(lx,ly,c='green')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker = 'o',s = 60, c = 'red')
plt.scatter(df_test_negaive['Clump Thickness'],df_test_negaive['Cell Size'],marker = 'x',s = 70, c = 'black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()

lr = LogisticRegression()
lr.fit(df_train[['Clump Thickness','Cell Size']],df_train['Type'])
print ('Testing accuracy (10 training samples):', lr.score(df_test[['Clump Thickness','Cell Size']],df_test['Type']))

intercept = lr.intercept_
coef = lr.coef_[0,:]  #多维数组
ly = (-intercept - lx * coef[0])/coef[1]

plt.plot(lx,ly,c='green')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker = 'o',s = 60, c = 'red')
plt.scatter(df_test_negaive['Clump Thickness'],df_test_negaive['Cell Size'],marker = 'x',s = 70, c = 'black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()