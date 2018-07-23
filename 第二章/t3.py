# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:07:46 2017
导入新闻抓取器，不同与之前的预存数据，fetch_20newsgroups需要即时从互联网下载数据,朴素贝叶斯进行分类
基于新闻数据转换为词向量
TF-IDF（词频-逆向文件频率）：如果某个词或短语在一篇文章中出现的频率TF高，并且在其他文章中很少出现，
则认为此词或者短语具有很好的类别区分能力，适合用来分类
@author: mjw
"""
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset = 'all')
print (len(news.data))
print (news.data[0])

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25,random_state=33)
#将文本转换为特征向量，利用朴树贝叶斯模型从训练数据中估计参数，
#最后利用概率参数对同样转换为特征向量的测试新闻样本进行类别预测
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
#选用多项式模型，特征为离散变量，TF-IDF符合该（分布）模型；高斯模型，GaussianNB（特征为连续变量），不做平滑处理；
#伯努利分布（二值分布）特征取值为0或1，BernoulliNB
mnb = MultinomialNB() #默认配置初始化
mnb.fit(X_train,y_train)
y_predict = mnb.predict(X_test)
from sklearn.metrics import classification_report
print ('Accuracy of Naive Bayes Classifier is:',mnb.score(X_test,y_test))
print (classification_report(y_test,y_predict,target_names=news.target_names)) 