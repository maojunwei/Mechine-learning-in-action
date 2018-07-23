# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:58:55 2018
使用内置的LinearRegressor,DNN以及Scikit-learn中的集成回归模型对“美国波士顿房价数据进行回归预测”
skflow已整合到tensorflow中
@author: mjw
"""
from sklearn import datasets,metrics,preprocessing,cross_validation
boston = datasets.load_boston()
x,y = boston.data,boston.target
x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size=0.25,random_state=33)
scaler = preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

import tensorflow.contrib.learn as learn  #替换
#tf_lr = learn.LinearRegressor(steps = 10000,learning_rate=0.01,batch_size=50)
tf_lr = learn.LinearRegressor(feature_columns = [])
tf_lr.fit(x_train,y_train)
tf_lr_y_predict = tf_lr.predict(x_test) 

print('--------Tensorflow linear regressor on boston dataset--------')
print('MAE',metrics.mean_absolute_error(tf_lr_y_predict,y_test))   #平均绝对误差
print('MSE',metrics.mean_squared_error(tf_lr_y_predict,y_test))   #均方误差
print('R-squared value',metrics.r2_score(tf_lr_y_predict,y_test))