# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:39:17 2018
基于波士顿房价数据，采用多种回归方法进行
@author: mjw
"""
import sys  
import os  
import time  
from sklearn import metrics  
import numpy as np   
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.datasets import load_boston

def SGD_regressor(train_x, train_y):
    from sklearn.linear_model import SGDRegressor
    model = SGDRegressor()
    model.fit(X_train,y_train)
    return model
def LR_regressor(train_x, train_y):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train,y_train)
    return model
def linear_svr_regressor(train_x, train_y): #线性核函数
    from sklearn.svm import SVR
    model =SVR(kernel = 'linear')
    model.fit(X_train,y_train)
    return model
def poly_svr_regressor(train_x, train_y):  #多项式核函数
    from sklearn.svm import SVR
    model =SVR(kernel = 'poly')
    model.fit(X_train,y_train)
    return model
def rbf_svr_regressor(train_x, train_y):  #径向基核函数
    from sklearn.svm import SVR
    model =SVR(kernel = 'rbf')
    model.fit(X_train,y_train)
    return model
def uni_knr_regressor(train_x, train_y): #weights = 'uniform'
    from sklearn.neighbors import KNeighborsRegressor
    model = KNeighborsRegressor(weights = 'uniform')
    model.fit(X_train,y_train)
    return model
def dis_knr_regressor(train_x, train_y): #weights = 'distance'，根据距离加权回归
    from sklearn.neighbors import KNeighborsRegressor
    model =KNeighborsRegressor(weights = 'distance')
    model.fit(X_train,y_train)
    return model
def decisiontree_regressor(train_x, train_y): #单一回归树
    from sklearn.tree import DecisionTreeRegressor
    model =DecisionTreeRegressor()
    model.fit(X_train,y_train)
    return model
def randomforests_regressor(train_x, train_y): #随机森林
    from sklearn.ensemble import RandomForestRegressor
    model =RandomForestRegressor()
    model.fit(X_train,y_train)
    return model
def extratrees_regressor(train_x, train_y): #极端回归森林
    from sklearn.ensemble import ExtraTreesRegressor
    model =ExtraTreesRegressor()
    model.fit(X_train,y_train)
    return model
def gradientboosting_regressor(train_x, train_y): #梯度提升回归树
    from sklearn.ensemble import GradientBoostingRegressor
    model =GradientBoostingRegressor()
    model.fit(X_train,y_train)
    return model
#uni_knr = KNeighborsRegressor(weights = 'uniform')
#dis_knr = KNeighborsRegressor(weights = 'distance')
#lr_y_predict = lr.predict(X_test)
#sgdr_y_predict = sgdr.predict(X_test)
if __name__ == "__main__":
    test_regressors = ['SGD','LR','SVR1','SVR2','SVR3','KNr1','KNr2','DT','RF','ERT','Gbt']
    regressors = {'SGD':SGD_regressor,
                   'LR':LR_regressor,
                   'SVR1':linear_svr_regressor,
                   'SVR2':poly_svr_regressor,
                   'SVR3':rbf_svr_regressor,
                   'KNr1':uni_knr_regressor,  
                   'KNr2':dis_knr_regressor,  
                   'DT':decisiontree_regressor,
                   'RF':randomforests_regressor,
                   'ERT':extratrees_regressor,
                   'Gbt':gradientboosting_regressor
            }
    boston = load_boston()
    X = boston.data
    y = boston.target
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 33,test_size = 0.25)
    ss_X = StandardScaler()
    ss_y = StandardScaler()
    X_train = ss_X.fit_transform(X_train)
    X_test = ss_X.transform(X_test)
    y_train = ss_y.fit_transform(y_train)
    y_test = ss_y.transform(y_test)  
    for regressor in test_regressors:
        print('***************%s***************'%regressor)
        start_time = time.time()
        model = regressors[regressor](X_train,y_train)
        print('trainng took %f s!'%(time.time() - start_time))
        y_predict = model.predict(X_test)
        print('R-squared:',model.score(X_test,y_test))
        print('MSE:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(y_predict)))
        print('MAE:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(y_predict)))
        

