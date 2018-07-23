# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 10:32:54 2017
基于泰坦尼克乘客数据，对比单一决策树、随机森林分类器、设置参数的随机森林分类器、梯度提升决策树
@author: mjw
"""
import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
X = titanic[['pclass','age','sex']]
y = titanic['survived']
X['age'].fillna(X['age'].mean(),inplace = True)

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 33)
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse = False)   #将dict类型中的list转换为numpy array
#将类别型（非数值）都单独剥离，生成各自一列特征，数值型的则保持不变
X_train = vec.fit_transform(X_train.to_dict(orient = 'record'))
#print (vec.feature_names_)
X_test = vec.fit_transform(X_test.to_dict(orient = 'record'))

#使用提供单一决策树进行模型训练以及预测分析
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc_y_pred = dtc.predict(X_test)

#使用随即森林分类器进行集成模型的训练以及预测分析
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_pred = rfc.predict(X_test)
"""
boostrap抽样方法，从训练集中随机可重复地选择n个样本，n值过大易导致随机森林的过拟合
超参：抽样的样本数量，以及节点划分使用的特征数量
"""
rfc1 = RandomForestClassifier(criterion = 'entropy',n_estimators = 10,random_state = 1,n_jobs = 2)
#使用10颗决策树构造随机森林，n_jobs参数设定训练过程所需内核的数量
rfc1.fit(X_train,y_train)
rfc_y_pred1 = rfc1.predict(X_test)

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_pred = gbc.predict(X_test)

from sklearn.metrics import classification_report
print ('Accuracy of decision tree is:',dtc.score(X_test,y_test))
print (classification_report(y_test,dtc_y_pred)) 
print ('Accuracy of random forest classifier is:',rfc.score(X_test,y_test))
print (classification_report(y_test,rfc_y_pred)) 
print ('Accuracy of random forest classifier with setting is:',rfc1.score(X_test,y_test))
print (classification_report(y_test,rfc_y_pred1)) 
print ('Accuracy of gradient boosting decision tree is:',gbc.score(X_test,y_test))
print (classification_report(y_test,gbc_y_pred)) 