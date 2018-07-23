# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:37:58 2017

@author: mjw
"""
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.datasets import load_boston
if __name__ == '__main__':   
    boston = load_boston()
    data = pd.read_csv('E:\Datasets\data.csv')
    data = np.array(data)
    X = boston.data
    y = boston.target
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 33,test_size = 0.25)
#    X_train,X_test,y_train,y_test = train_test_split(data[:,0:2],data[:,-1],random_state=33,test_size=0.25)
    ss_X = StandardScaler()
    ss_y = StandardScaler()
    X_train = ss_X.fit_transform(X_train)
    X_test = ss_X.transform(X_test)
    y_train = ss_y.fit_transform(y_train)
    y_test = ss_y.transform(y_test)
    #线性核函数
    linear_svr =SVR(kernel = 'linear')
    linear_svr.fit(X_train,y_train)
    linear_svr_y_predict = linear_svr.predict(X_test)
    #多项式核函数
    poly_svr =SVR(kernel = 'poly')
    poly_svr.fit(X_train,y_train)
    poly_svr_y_predict = poly_svr.predict(X_test)
    #径向基核函数
    rbf_svr =SVR(kernel = 'rbf')
    rbf_svr.fit(X_train,y_train)
    rbf_svr_y_predict = rbf_svr.predict(X_test)
    print('-------------The result of linear SVR-------------')     
    print('R-squared',linear_svr.score(X_test,y_test))
    print('MSE:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(linear_svr_y_predict)))
    print('MAE:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(linear_svr_y_predict)))
    print('-------------The result of poly SVR-------------')     
    print('R-squared',poly_svr.score(X_test,y_test))
    print('MSE:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(poly_svr_y_predict)))
    print('MAE:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(poly_svr_y_predict)))
    print('-------------The result of rbf SVR-------------')     
    print('R-squared',rbf_svr.score(X_test,y_test))
    print('MSE:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rbf_svr_y_predict)))
    print('MAE:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rbf_svr_y_predict)))