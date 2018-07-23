# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:29:16 2018
Tensorflow框架,入门学习
@author: mjw
"""
#import tensorflow as tf
#import numpy as np
#greeting = tf.constant('Hello Google Tensorflow! ')
#sess = tf.Session()
#result = sess.run(greeting)
#print(result)
#sess.close()
#import tensorflow as tf
#import numpy as np
#matrixl1 = tf.constant([[3.,3.]])
#matrixl2 = tf.constant([[2.],[2.]])
#product=tf.matmul(matrixl1,matrixl2) #两个算子相乘，作为新算例
#linear=tf.add(product,tf.constant(2.0))#与标量拼接，作为最终算例
#with tf.Session() as sess:     #将上面所有的单独算例拼接成流程图来执行
#    result = sess.run(linear)
#    print(result)
'''
使用Tensorflow自定义一个线性分类器
'''    
import tensorflow as tf
import numpy as np
import pandas as pd
train = pd.read_csv('E:/Datasets/Breast-Cancer/breast-cancer-train.csv')
test = pd.read_csv('E:/Datasets/Breast-Cancer/breast-cancer-test.csv')
X_train = np.float32(train[['Clump Thickness','Cell Size']].T)
y_train = np.float32(train['Type'].T)
X_test = np.float32(test[['Clump Thickness','Cell Size']].T)
y_test = np.float32(test['Type'].T)

#定义一个tensorflow的变量b作为线性模型的截距，设置初始值1.0
b=tf.Variable(tf.zeros([1]))
#定义变量w作为线性模型的系数，并设置初始值为-1.0至1.0之间均匀分布的随机数
W = tf.Variable(tf.random_uniform([1,2],-1.0,1.0)) 

#显式定义这个线性函数
y=tf.matmul(W,X_train) + b
loss = tf.reduce_mean(tf.square(y-y_train))
#使用梯度下降估计参数W，b,并设置迭代步长为0.01
optimizer = tf.train.GradientDescentOptimizer(0.01)

train = optimizer.minimize(loss)   #最小二乘损失为优化目标
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
#迭代1000轮次，训练参数
for step in range(0,1000):
    sess.run(train)
    if step%200 == 0:
        print(step,sess.run(W),sess.run(b))

test_negative = test.loc[test['Type'] == 0][['Clump Thickness','Cell Size']]
test_positive = test.loc[test['Type'] == 1][['Clump Thickness','Cell Size']]

#import matplotlib.pyplot as plt
#plt.scatter(test_negative['Clump Thickness'],test_negative['Cell Size'],marker = 'o',s=200,c='red')
#plt.scatter(test_positive['Clump Thickness'],test_positive['Cell Size'],marker = 'x',s=150,c='black')
#plt.xlabel('Clump Thickness')
#plt.ylabel('Cell Size')
#lx = np.arange(0,12)
#ly = (0.5-sess.run())
        
        