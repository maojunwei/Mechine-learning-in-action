# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:01:29 2017
@author: mjw
基于lris数据集进行knn分类实验
"""
from sklearn.datasets import load_iris
import numpy as np
iris = load_iris()
#iris.data.shape
'''
查看数据说明
'''
print (iris.DESCR)
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.25,random_state=33)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


knc1 = KNeighborsClassifier(n_neighbors=5,p=1,metric='minkowski')#近邻k值，p=1曼哈顿距离，p=2欧氏距离
knc.fit(X_train,y_train)
knc1.fit(X_train,y_train)
y_predict = knc.predict(X_test)
y_predict1 = knc1.predict(X_test)
from sklearn.metrics import classification_report
print ('Accuracy of K-Nearest Neighbor Classifier is:',knc.score(X_test,y_test))
print (classification_report(y_test,y_predict,target_names=iris.target_names)) 

print ('Accuracy of K-Nearest Neighbor Classifier with setting is:',knc.score(X_test,y_test))
print (classification_report(y_test,y_predict1,target_names=iris.target_names)) 
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
#def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
#
#    # setup marker generator and color map
#    markers = ('s', 'x', 'o', '^', 'v')
#    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
#    cmap = ListedColormap(colors[:len(np.unique(y))])
#
#    # plot the decision surface
#    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
#                           np.arange(x2_min, x2_max, resolution))
#    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#    Z = Z.reshape(xx1.shape)
#    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
#    plt.xlim(xx1.min(), xx1.max())
#    plt.ylim(xx2.min(), xx2.max())
#
#    for idx, cl in enumerate(np.unique(y)):
#        plt.scatter(x=X[y == cl, 0], 
#                    y=X[y == cl, 1],
#                    alpha=0.8, 
#                    c=colors[idx],
#                    marker=markers[idx], 
#                    label=cl, 
#                    edgecolor='black')
#
#    # highlight test samples
#    if test_idx:
#        # plot all samples
#        X_test, y_test = X[test_idx, :], y[test_idx]
#
#        plt.scatter(X_test[:, 0],
#                    X_test[:, 1],
#                    c='',
#                    edgecolor='black',
#                    alpha=1.0,
#                    linewidth=1,
#                    marker='o',
#                    s=100, 
#                    label='test set')
#X_combined = np.vstack((X_train[0:2,:],X_test[0:2,:])) #水平(按列顺序)把数组给堆叠起来
##y_combined = np.hstack((y_train[0:2,:],y_test[0:2,:]))  #垂直(按行顺序)把数组给堆叠起来
#plot_decision_regions(X_combined,y_combined,classifier=knc)