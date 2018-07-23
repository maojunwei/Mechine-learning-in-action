# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:49:31 2018

@author: mjw
"""
'''
并行搜索  由于各个新模型在执行交叉验证的过程中间是相互独立的，可以利用多核处理器/分布式的计算资源经费选哪个并行搜索，成倍地节省运算时间
'''
from sklearn.datasets import fetch_20newsgroups
import numpy as np
news = fetch_20newsgroups(subset = 'all')
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(news.data[:3000],news.target[:3000],test_size = 0.25,random_state=33)
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
clf = Pipeline([('vect',TfidfVectorizer(stop_words = 'english',analyzer = 'word')),('svc',SVC())])
parameters = {'svc__gamma':np.logspace(-2,1,4),'svc__C':np.logspace(-1,1,3)}
from sklearn.grid_search import GridSearchCV #网格搜索模块
gs = GridSearchCV(clf,parameters,verbose = 2,refit = True,cv = 3,n_jobs=-1)  #n_jobs=-1代表使用该计算机全部的CPU
time_ = gs.fit(x_train,y_train)
print(gs.best_params_,gs.best_score_)
print(gs.score(x_test,y_test))
                 