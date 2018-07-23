# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:03:30 2018
模型验证：交叉验证、网格搜索,并行搜索
@author: mjw
"""
'''
使用单线程对文本分类的支持向量机的超参数组合执行网格搜索
'''
'''
SVC参数解释  
（1）C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0；  
（2）kernel：参数选择有RBF, Linear, Poly, Sigmoid, 默认的是"RBF";  
（3）degree：if you choose 'Poly' in param 2, this is effective, degree决定了多项式的最高次幂；  
（4）gamma：核函数的系数('Poly', 'RBF' and 'Sigmoid'), 默认是gamma = 1 / n_features;  
（5）coef0：核函数中的独立项，'RBF' and 'Poly'有效；  
（6）probablity: 可能性估计是否使用(true or false)；  
（7）shrinking：是否进行启发式；  
（8）tol（default = 1e - 3）: svm结束标准的精度;  
（9）cache_size: 制定训练所需要的内存（以MB为单位）；  
（10）class_weight: 每个类所占据的权重，不同的类设置不同的惩罚参数C, 缺省的话自适应；  
（11）verbose: 跟多线程有关，不大明白啥意思具体；  
（12）max_iter: 最大迭代次数，default = 1， if max_iter = -1, no limited;  
（13）decision_function_shape ： ‘ovo’ 一对一, ‘ovr’ 多对多  or None 无, default=None  
（14）random_state ：用于概率估计的数据重排时的伪随机数生成器的种子。  
'''
from sklearn.datasets import fetch_20newsgroups#新闻抓取器
import numpy as np
news = fetch_20newsgroups(subset = 'all')
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(news.data[:3000],news.target[:3000],test_size = 0.25,random_state = 33)
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
clf = Pipeline([('vect',TfidfVectorizer(stop_words = 'english',analyzer = 'word')),('svc',SVC())]) #使用pipeline简化系统搭建流程，将文本抽取与分类模型串联起来
parameters = {'svc__gamma':np.logspace(-2,1,4),'svc__C':np.logspace(-1,1,3)} #logspace创建等比数列，默认10为底；核函数系数(4),惩罚系数C(3),超参组合12组
from sklearn.grid_search import GridSearchCV #网格搜索模块
gs = GridSearchCV(clf,parameters,verbose = 2,refit = True,cv = 3) #refit = True,得到的最佳超参数直接用于模型

#执行单线程网格搜索
time_ =gs.fit(x_train,y_train)
print(gs.best_params_,gs.best_score_)
print(gs.score(x_test,y_test))