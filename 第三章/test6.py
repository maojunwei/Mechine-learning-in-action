# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 21:01:13 2018
自然语言处理包(NLTK),词向量(Word2Vec)技术,XGboost模型
@author: mjw
"""
'''
词袋法，文本向量化表示，对文本之间在内容的相似性进行一定程度的度量
量化方法：CountVectorizer,TfidfVectorizer
'''
#使用词袋法（Bag-of-Words)对示例文本进行特征向量化
sent1 = 'The cat is walking in the bedroom.'
sent2 = 'A dog was running across the kitchen.'
from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()
sentences = [sent1,sent2]
#print(count_vec.fit_transform(sentences).toarray()) #输出特征量化后的表示
#print(count_vec.get_feature_names()) #输出向量各个维度的特征含义

#import nltk   #nltk
#sens = nltk.sent_tokenize(sent1)
#print(sens)
#tokens_1 = nltk.word_tokenize(sent1)
'''
词向量技术，
'''
from sklearn.datasets import fetch_20newsgroups
import numpy as np
news = fetch_20newsgroups(subset = 'all')
x,y = news.data,news.target
from bs4 import BeautifulSoup
import nltk,re
def news_to_sentences(news):             #将每条新闻中的句子逐一剥离出来，并返回一个句子的列表
    news_text = BeautifulSoup(news).get_text()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(news_text)
    sentences = []
    for sent in raw_sentences:
        sentences.append(re.sub('[^a-ZA-Z]',' ',sent.lower().strip()).split())
    return sentences
sentences = []
for i in x:
    sentences += news_to_sentences(i)

from gensim.models import word2vec
num_features = 300      #词向量维度
min_word_count = 20     #保证被考虑的词汇频度
num_workers = 2
context = 5
downsampling = 1e-3

from gensim.models import word2vec
model = word2vec.Word2Vec(sentences,workers = num_workers,size = num_features,min_count = min_word_count,window = context,sample = downsampling)
model.init_sims(replace = True)
model.most_similar('morning')