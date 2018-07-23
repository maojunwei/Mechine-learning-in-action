# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:53:08 2018

@author: mjw
"""
import pandas as pd
import numpy as np
digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',header=None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',header=None)
X_digits = digits_train[np.arange(64)]
y_digits = digits_train[64]
from sklearn.decomposition import PCA
estimator = PCA(n_components=2)
X_pca = estimator.fit_transform(X_digits)
from matplotlib import pyplot as plt
def plot_pca_scatter():
    colors = ['black','blue','purple','yellow','white','red','lime','cyan','orange','gray']
    
    for i in xrange(len(colors)):
        px = X_pca[:,0][y_digits.as_matrix() == i]
        py = X_pca[:,1][y_digits.as_matrix() == i]
        plt.scatter(px,py,c = colors[i])
        
    plt.legend(np.arange(0,10).astype(str))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second P rincipal Component')
    plt.show()
    
#plt_pca_scatter()

X_train = digits_train[np.arange(64)]
y_train = digits_train[64]
X_test = digits_test[np.arange(64)]
y_test = digits_test[64]
from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(X_train,y_train)
y_predict = svc.predict(X_test)

estimator = PCA(n_components = 20)
pca_X_train = estimator.fit_transform(X_train)
pca_X_test = estimator.transform(X_test)

pca_svc = LinearSVC()
pca_svc.fit(pca_X_train,y_train)
pca_y_predict = pca_svc.predict(pca_X_test)

from sklearn.metrics import classification_report
print(svc.score(X_test,y_test))
print (classification_report(y_test,pca_y_predict,target_names = np.arange(10).astype(str)))
print(pca_svc.score(pca_X_test,y_test))
print(classification_report(y_test,pca_y_predict,target_names = np.arange(10).astype(str)))