# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:01:22 201
基于波士顿数据采用k近邻
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
from sklearn.neighbors import KNeighborsRegressor
#初始化k近邻回归器，并且调整配置，使得预测的方式为平均回归：weights = 'uniform'
uni_knr = KNeighborsRegressor(weights = 'uniform')
uni_knr.fit(X_train,y_train)
uni_knr_y_predict = uni_knr.predict(X_test)
#初始化k近邻回归器，并且调整配置，使得预测的方式为根据距离加权回归：weights = 'distance'
dis_knr = KNeighborsRegressor(weights = 'distance')
dis_knr.fit(X_train,y_train)
dis_knr_y_predict = dis_knr.predict(X_test)
from sklearn.preprocessing import StandardScaler
#对特征及目标值均归一化
ss_X = StandardScaler()
ss_y = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print('-------------The result of uniform-weighted KNeighborsRegressor-------------')     
print('R-squared',r2_score(y_test,uni_knr_y_predict))
print('MSE:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(uni_knr_y_predict)))
print('MAE:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(uni_knr_y_predict)))

