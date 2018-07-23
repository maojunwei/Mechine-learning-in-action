# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 18:35:17 2018

@author: mjw
"""
'''
DictVectorizer对字典存储的数据进行特征抽取与向量化
'''
measurements =[{'city':'Dubai','temperature':33.},{'city':'london','temperature':12.},{'city':'San Fransisco','temperature':18.}]
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
print(vec.fit_transform(measurements).toarray())
print(vec.get_feature_names())

from sklearn.datasets import fetch_20newsgroups #导入20类新闻文本抓取器
news = fetch_20newsgroups(subset = 'all')
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(news.data,news.target,test_size = 0.25,random_state = 33)

from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()    #只使用词频统计的方式将原始训练和测试文本转化为特征向量
X_count_train = count_vec.fit_transform(X_train)
X_count_test = count_vec.transform(X_test)

from sklearn.naive_bayes import MultinomialNB #多项式模型
mnb_count = MultinomialNB()
mnb_count.fit(X_count_train,y_train)

print('The accuracy of classifying 20newsgroups using Naive Bayes(CountVectorizer without filtering stopwords):',mnb_count.score(X_count_test,y_test))
y_count_predict = mnb_count.predict(X_count_test)

from sklearn.metrics import classification_report
print (classification_report(y_test,y_count_predict,target_names = news.target_names))

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer()      #使用tfidf的方式，将原始训练和测试文本转化为特征向量
X_tfidf_train = tfidf_vec.fit_transform(X_train)
X_tfidf_test = tfidf_vec.transform(X_test)

mnb_tfidf = MultinomialNB()
mnb_tfidf.fit(X_tfidf_train,y_train) 

print('The accuracy of classifying 20newsgroups using Naive Bayes(TfidfVectorizer without filtering stopwords):',mnb_tfidf.score(X_tfidf_test,y_test))
y_tfidf_predict = mnb_tfidf.predict(X_tfidf_test)
print (classification_report(y_test,y_tfidf_predict,target_names = news.target_names))

count_filter_vec,tfidf_filter_vec = CountVectorizer(analyzer = 'word',stop_words = 'english'),TfidfVectorizer(analyzer = 'word',stop_words = 'english')
X_count_filter_train = count_filter_vec.fit_transform(X_train) #使用带有停用词过滤的CountVectorizer进行量化处理
X_count_filter_test = count_filter_vec.transform(X_test)

X_tfidf_filter_train = tfidf_filter_vec.fit_transform(X_train) #使用带有停用词过滤的TfidfVectorizer进行量化处理
X_tfidf_filter_test = tfidf_filter_vec.transform(X_test)

mnb_count_filter = MultinomialNB()
mnb_count_filter.fit(X_count_filter_train,y_train)
print('The accuracy of classifying 20newsgroups using Naive Bayes(CountVectorizer by filtering stopwords):',mnb_count_filter.score(X_count_filter_test,y_test))
y_count_filter_predict = mnb_count_filter.predict(X_count_filter_test)

mnb_tfidf_filter = MultinomialNB()
mnb_tfidf_filter.fit(X_tfidf_filter_train,y_train)
print('The accuracy of classifying 20newsgroups using Naive Bayes(TfidfVectorizer by filtering stopwords):',mnb_tfidf_filter.score(X_tfidf_filter_test,y_test))
y_tfidf_filter_predict = mnb_tfidf_filter.predict(X_tfidf_filter_test)

from sklearn.metrics import classification_report
print (classification_report(y_test,y_count_filter_predict,target_names = news.target_names))
print (classification_report(y_test,y_tfidf_filter_predict,target_names = news.target_names))
