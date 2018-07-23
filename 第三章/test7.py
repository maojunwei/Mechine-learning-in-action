# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 21:05:41 2018
XGBoost模型
@author: mjw
"""
import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
x = titanic[['pclass','age','sex']]
y = titanic['survived']

x['age'].fillna(x['age'].mean(),inplace = True)
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 33)
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse = False)
X_train = vec.fit_transform(X_train.to_dict(orient = 'record')) #将DataFrame格式的数据转换为字典形式
X_test = vec.transform(X_test.to_dict(orient = 'record'))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
print('The accuracy of random forest classifier on testing set:',rfc.score(X_test,y_test))

from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(X_train,y_train)
print('The accuracy of extreme gradient boosting classifier on testing set:',xgbc.score(X_test,y_test))