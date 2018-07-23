# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 08:45:08 2017
基于决策树进行泰坦尼克乘客数据查验
@author: mjw
"""
import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
titanic.head()

X = titanic[['pclass','age','sex']]
y = titanic['survived']
"""
对当前选择的特征进行探查
"""
#X.info()
X['age'].fillna(X['age'].mean(),inplace = True)
'''
对补充完的数据进行检查
'''
#X.info()

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 33)
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse = False)   #将dict类型中的list转换为numpy array
#将类别型（非数值）都单独剥离，生成各自一列特征，数值型的则保持不变
X_train = vec.fit_transform(X_train.to_dict(orient = 'record'))
#print (vec.feature_names_)
X_test = vec.fit_transform(X_test.to_dict(orient = 'record'))

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc1 = DecisionTreeClassifier(criterion = 'entropy',max_depth = 6,random_state = 0)
dtc.fit(X_train,y_train)
dtc1.fit(X_train,y_train)
y_predict = dtc.predict(X_test)
y_predict1 = dtc1.predict(X_test)
from sklearn.metrics import classification_report
print ('Accuracy of decision tree is:',dtc.score(X_test,y_test))
print (classification_report(y_test,y_predict,target_names=['died','survived'])) print ('Accuracy of decision tree with setting is:',dtc1.score(X_test,y_test))
print (classification_report(y_test,y_predict1,target_names=['died','survived']))
#sklearn.metrics.classification_report(y_true, y_pred, labels=None, target_names=None,sample_weight=None, digits=2)