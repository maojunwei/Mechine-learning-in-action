# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 20:54:57 2018

@author: mjw
"""
import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
y = titanic['survived']
X = titanic.drop(['row.names','name','survived'], axis = 1)   #删除数据的几列

X['age'].fillna(X['age'].mean(),inplace = True)
X.fillna('UNKNOWN',inplace = True)

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 33)
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
X_train = vec.fit_transform(X_train.to_dict(orient = 'record')) #将DataFrame格式的数据转换为字典形式
X_test = vec.transform(X_test.to_dict(orient = 'record'))
'''
DataFrame.to_dict(orient='dict')
将DataFrame格式的数据转化成字典形式
参数：当然参数orient可以是字符串{'dict', 'list', 'series', 'split', 'records', 'index'}中的任意一种来决定字典中值的类型
字典dict（默认）：类似于{列：{索引：值}}这样格式的字典
列表list：类似于{列：[值]}这种形式的字典
序列series：类似于{列：序列（值）}这种形式的字典
分解split：类似于{索引：[索引]，列：[列]，数据：[值]}这种形式的字典
记录records：类似于[{列：值}，...，{列：值}]这种形式的列表
索引index：类似于{索引：{列：值}}这种形式的字典
在新版本0.17.0中，允许缩写，s表示序列，sp表示分裂
返回：结果：像{列：{索引：值}}这种形式的字典
'''
#print (len(vec.feature_names_)) #474
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = 'entropy')
dt.fit(X_train,y_train)
print(dt.score(X_test,y_test))

'''
导入特征筛选器,筛选前20%的特征
'''
from sklearn import feature_selection
#selectKBest:选择排名前n个；SelectPercentile，前n%个
fs = feature_selection.SelectPercentile(feature_selection.chi2,percentile = 20)
X_train_fs = fs.fit_transform(X_train,y_train)
dt.fit(X_train_fs,y_train)
X_test_fs = fs.transform(X_test)
print(dt.score(X_test_fs,y_test))
"""
交叉验证，按照固定间隔的百分比筛选特征，并作图展示性能随特征筛选比例的变化
"""
from sklearn.cross_validation import cross_val_score
import numpy as np
percentiles = range(1,100,2)
results = []

for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2,percentile = i)
    X_train_fs = fs.fit_transform(X_train,y_train)
    scores = cross_val_score(dt,X_train_fs,y_train,cv = 5)
    results = np.append(results,scores.mean())
print (results)

opt = np.where(results == results.max())[0]
print('Optimal number of feature %d' %percentiles[opt])

import pylab as pl
pl.plot(percentiles,results)
pl.xlabel('percentiles of features')
pl.ylabel('accuracy')
pl.show()

from sklearn import feature_selection
fs = feature_selection.SelectPercentile(feature_selection.chi2,percentile = 7)
X_train_fs = fs.fit_transform(X_train,y_train)
dt.fit(X_train_fs,y_train)
X_test_fs = fs.transform(X_test)
print(dt.score(X_test_fs,y_test))

    