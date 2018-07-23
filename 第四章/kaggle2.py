# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:35:25 2018

@author: mjw
"""
import pandas as pd
train = pd.read_csv('E:/Datasets/IMDB/labeledTrainData.tsv',delimiter='\t')
test = pd.read_csv('E:/Datasets/IMDB/testData.tsv',delimiter='\t')
#train.head()
#test.head()
from bs4 import BeautifulSoup  #整洁原始文本
import re
from nltk.corpus import stopwords

#完成对原始评论的三项数据预处理任务
def review_to_text(review,remove_stopwords):
    raw_text = BeautifulSoup(review,'html').get_text()  #去掉html标记
    letters = re.sub('[^a-zA-z]',' ',raw_text)          #去掉非字母字符
    words = letters.lower().split()
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
    return words

X_train = []
for review in train['review']:
    X_train.append(' '.join(review_to_text(review,True)))
X_test = []
for review in test['review']:
    X_test.append(' '.join(review_to_text(review,True)))
y_train = train['sentiment']

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

#使用Pipelie搭建两组使用朴素贝叶斯模型的分类器，分别使用不同的特征量化方法
pip_count = Pipeline([('count_vec',CountVectorizer(analyzer = 'word')),('mnb',MultinomialNB())])
pip_tfidf = Pipeline([('tfidf_vec',TfidfVectorizer(analyzer = 'word')),('mnb',MultinomialNB())])
#分别配置用于模型超参数搜索的组合

params_count = {'count_vec__binary':[True,False],'count_vec__ngram_range':[(1,1),(1,2)],'mnb__alpha':[0.1,1.0,10.0]}
params_tfidf = {'tfidf_vec__binary':[True,False],'tfidf_vec__ngram_range':[(1,1),(1,2)],'mnb__alpha':[0.1,1.0,10.0]}

gs_count = GridSearchCV(pip_count,params_count,cv=4,verbose=1)
gs_count.fit(X_train,y_train)

print(gs_count.best_score_)  #输出交叉验证中最佳的准确性得分以及超参组合
print(gs_count.best_params_)

#以最佳的超参数组合配置模型并对尝试数据进行预测
count_y_predict = gs_count.predict(X_test)
gs_tfidf = GridSearchCV(pip_tfidf,params_tfidf,cv=4,verbose=1)
gs_tfidf.fit(X_train,y_train)

print(gs_tfidf.best_score_)  #输出交叉验证中最佳的准确性得分以及超参组合
print(gs_tfidf.best_params_)
tfidf_y_predict = gs_tfidf.predict(X_test)

#使用pandas对需要提交的数据进行格式化
submission_count = pd.DataFrame({'id':test['id'],'sentiment':count_y_predict})
submission_tfidf = pd.DataFrame({'id':test['id'],'sentiment':tfidf_y_predict})

submission_count.to_csv('F:/dataset/submission_count.csv',index=False)
submission_tfidf.to_csv('F:/dataset/submission_tfidf.csv',index=False)
"""
读入未标记数据
"""
unlabeled_train = pd.read_csv('E:/Datasets/IMDB/unlabeledTrainData.tsv',delimiter = '\t',quoting=3)
import nltk.data
tokenizer = nltk.data.load()

