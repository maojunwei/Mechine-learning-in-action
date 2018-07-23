# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 08:37:34 2018
使用线性回归模型拟合实验，L1\L2正则实验
@author: mjw
"""
x_train = [[6],[8],[10],[14],[18]]
y_train = [[7],[9],[13],[17.5],[18]]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

import numpy as np
xx = np.linspace(0,26,100)#0-25均匀采样100个数据点
xx = xx.reshape(xx.shape[0],1) #转置
yy = regressor.predict(xx)

import matplotlib.pyplot as plt
plt.scatter(x_train,y_train)
plt1, = plt.plot(xx,yy,label = 'Degree = 1')
plt.axis([0,25,0,25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles = [plt1])
plt.show()
print('The R-squared value of linear regressor performing on the training data is',regressor.score(x_train,y_train))

from sklearn.preprocessing import PolynomialFeatures#多项式特征产生器 1,x,x^2
poly2 = PolynomialFeatures(degree = 2)
x_train_poly2 = poly2.fit_transform(x_train)

regressor_poly2 = LinearRegression()

regressor_poly2.fit(x_train_poly2,y_train)
xx_poly2 = poly2.transform(xx)
yy_poly2 = regressor_poly2.predict(xx_poly2)
plt.scatter(x_train,y_train)

plt1, = plt.plot(xx,yy,label = 'Degree = 1')
plt2, = plt.plot(xx,yy_poly2,label = 'Degreee = 2')

plt.axis([0,25,0,25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles = [plt1,plt2])
plt.show()
print('The R-squared value of polynominal regressor(degree = 2) performing on the training data is:',regressor_poly2.score(x_train_poly2,y_train))

from sklearn.preprocessing import PolynomialFeatures
poly4 = PolynomialFeatures(degree = 4)
X_train_poly4 = poly4.fit_transform(x_train)

regressor_poly4 = LinearRegression()
regressor_poly4.fit(X_train_poly4,y_train)

xx_poly4 = poly4.transform(xx)
yy_poly4 = regressor_poly4.predict(xx_poly4)
plt.scatter(x_train,y_train)
plt1, = plt.plot(xx,yy,label = 'Degree=1')
plt2, = plt.plot(xx,yy_poly2,label = 'Degree=2')
plt4, = plt.plot(xx,yy_poly4,label = 'Degree=4')
plt.axis([0,25,0,25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles = [plt1,plt2,plt4])
plt.show()
print('The R-squared value of polynominal regressor(degree = 4) performing on the training data is:',regressor_poly4.score(X_train_poly4,y_train))

x_test = [[6],[8],[11],[16]]
y_test = [[8],[12],[15],[18]]
print('1',regressor.score(x_test,y_test))
x_test_poly2 = poly2.transform(x_test)
print('2',regressor_poly2.score(x_test_poly2,y_test))

x_test_poly4 = poly4.transform(x_test)
print('4',regressor_poly4.score(x_test_poly4,y_test))

'''
正则化
'''
from sklearn.linear_model import Lasso     #L1
lasso_poly4 = Lasso()
lasso_poly4.fit(X_train_poly4,y_train)

from sklearn.linear_model import Ridge     #L2
ridge_poly4 = Ridge()
ridge_poly4.fit(X_train_poly4,y_train)

print('4+L1',lasso_poly4.score(x_test_poly4,y_test))  #x^4,^3..由高到低
print('4+L2',ridge_poly4.score(x_test_poly4,y_test))  #x^4,^3..由高到低
print('4+L1:coef',lasso_poly4.coef_)
print('4+L2:coef',ridge_poly4.coef_)
print('4:coef',regressor_poly4.coef_)