# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 09:58:39 2018
线性回归器
@author: mjw
"""
from sklearn.datasets import load_boston
boston = load_boston()
#print (boston.DESCR)
from sklearn.cross_validation import train_test_split
import numpy as np
X = boston.data
y = boston.target
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 33,test_size = 0.25)
#分析目标值的差异
print("The max target value is",np.max(boston.target))
print("The min target value is",np.min(boston.target))
print("The average target value is",np.mean(boston.target))
from sklearn.preprocessing import StandardScaler
#对特征及目标值均归一化
ss_X = StandardScaler()
ss_y = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
lr_y_predict = lr.predict(X_test)

from sklearn.linear_model import SGDRegressor
sgdr = SGDRegressor()
sgdr.fit(X_train,y_train)
sgdr_y_predict = sgdr.predict(X_test)

print('The value of default measurement of LinearRegression is',lr.score(X_test,y_test))#LinearRegression自带的评估模块

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print('-------------The result of LinearRegression-------------')     
print('R-squared',r2_score(y_test,lr_y_predict))
print('MSE:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))
print('MAE:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))

print('The value of default measurement of LinearRegression is',sgdr.score(X_test,y_test))#SGDRegression自带的评估模块

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print('-------------The result of sgdrRegression-------------')     
print('R-squared',r2_score(y_test,sgdr_y_predict))
print('MSE:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdr_y_predict)))
print('MAE:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdr_y_predict)))