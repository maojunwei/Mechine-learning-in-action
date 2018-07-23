# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 22:53:38 2018
k-means算法在手写数字图像数据上的使用
@author: mjw
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',header=None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',header=None)
X_train = digits_train[np.arange(64)]
y_train = digits_train[64]
X_test = digits_test[np.arange(64)]
y_test = digits_test[64]

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 10)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_test)
from sklearn import metrics
print(metrics.adjusted_rand_score(y_test,y_pred))#ARI进行聚类性能评估

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

plt.subplot(3,2,1)#分割出6个子图，并在1号子图作画
x1 = np.array([1,2,3,1,5,6,5,5,6,7,8,9,7,9])
x2 = np.array([1,3,2,2,8,6,7,6,7,1,2,1,1,3])
X = np.array(zip(x1,x2)).reshape(2,len(x1)

plt.xlim([0,10])
plt.ylim([0,10])
plt.title('Instances')
plt.scatter(x1,x2)

colors = ['b','g','r','c','m','y','k','b']
markers = ['o','s','D','v','^','p','*','+']

clusters = [2,3,4,5,8]
subplot_counter = 1
sc_scores = []
for t in clusters:
    subplot_counter += 1
    plt.subplot(3,2,subplot_counter)
    kmeans_model = KMeans(n_clusters=t).fit(X)
    
    for i,l in enumerate(kmeans_model.labels_):
        plt.plot(x1[i],x2[i],color=colors[l],marker=markers[l],ls='None')
        
        
    plt.xlim([0,10])
    plt.ylim([0,10])
    sc_score = silhouette_score(X,kmeans_model.label_,metric='euclidean')
    sc_scores.append(sc_score)
    plt.title('K=%s, silhouette coefficient = %0.03f'%(t,sc_score))
    
plt.figure()
plt.plot(clusters,sc_scores,'*-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient Score')

plt.show()
    

