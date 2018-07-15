# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 21:14:58 2017

@author: 001
"""
from numpy import *
import sys
#词表到向量的转换函数
def loaddataset():
    postinglist = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid' ]]
    classvec = [0,1,0,1,0,1]#标签,1代表侮辱文字，0代表正常
    return postinglist,classvec
#以列表读取文档中不重复词
def createvocablist(dataset):
    vocabset = set([])
    for document in dataset:
        vocabset = vocabset | set(document)
    return list(vocabset)
#词向量
def setofword2vec(vocablist,inputset):
    returnvec = [0]*len(vocablist)
    for word in inputset:
        if word in vocablist:
            returnvec[vocablist.index(word)] = 1
        else: print( "the word: %s is not in my vocabulary!" % word)
    return returnvec
#朴素贝叶斯分类器的训练函数(修改版)
def trainnb0(trainmatrix,traincategory):
    numtraindocs = len(trainmatrix)
    numwords = len(trainmatrix[0])
    pabusive = sum(traincategory)/float(numtraindocs)
    p0num = ones(numwords);  p1num = ones(numwords)
    p0denom = 2.0;   p1denom = 2.0
    for i in range(numtraindocs):
        if traincategory[i] == 1:
            p1num += trainmatrix[i]
            p1denom += sum(trainmatrix[i])
        else:
            p0num += trainmatrix[i]
            p0denom += sum(trainmatrix[i])
    p1vect = log(p1num/p1denom)
    p0vect = log(p0num/p0denom)
    return p0vect,p1vect,pabusive

def classifynb(vec2classify,p0vec,p1vec,pclass1):
    p1 = sum(vec2classify * p1vec) + log(pclass1)
    p0 = sum(vec2classify * p0vec) + log(1.0 - pclass1)
    if p1 > p0:
        return 1
    else:
        return 0
#朴素贝叶斯词袋模型
def bagofwords2vecmn(vocablist,inputset):
    returnvec = [0]*len(vocablist)
    for word in inputset:
        if word in vocablist:
            returnvec[vocablist.index(word)] += 1
    return returnvec
#朴素贝叶斯分类函数
def testingnb():
    listoposts,listclasses = loaddataset()
    myvocablist = createvocablist(listoposts)
    trainmat = []
    for post in listoposts:
        trainmat.append(bayes.setofword2vec(myvocablist,post))
    p0v,p1v,pab = trainnb0(trainmat,listclasses)
    testentry = ['love','my','dalmation']
    thisdoc = array(setofword2vec(myvocablist,testentry))
    print (testentry,'classified as:',classifynb(thisdoc,p0v,p1v,pab))
    testentry = ['stupid','garbage']
    thisdoc = array(setofword2vec(myvocablist,testentry))
    print (testentry,'classified as',classifynb(thisdoc,p0v,p1v,pab))
#文件解析及完整的垃圾邮件测试函数
def textparse(bigstring):   #将大字符串解析为字符串列表
    import re
    listoftokens = re.split(r'\W*',bigstring)
    return [tok.lower() for tok in listoftokens if len(tok) > 2]

def spamtest():
    doclist = []; classlist = []; fulltext = []
    for i in range(1,26):
        wordlist = textparse(open('email/spam/%d.txt' % i).read())
        doclist.append(wordlist)
        fulltext.extend(wordlist)
        classlist.append(1)
        wordlist = textparse(open('email/ham/%d.txt' % i).read())
        doclist.append(wordlist)
        fulltext.extend(wordlist)
        classlist.append(0)
    vocablist = createvocablist(doclist)
    trainingset = list(range(50)); testset=[]
    for i in range(10):
        randindex = int(random.uniform(0,len(trainingset)))
        testset.append(trainingset[randindex])
        del(trainingset[randindex])
    trainmat = [];   trainclasses = []
    for docindex in trainingset:
        trainmat.append(setofwords2vec(vocablist,doclist[docindex]))
        trainclasses.append(classlist[docindex])
    p0v,p1v,pspam = trainnb0(array(trainmat),array(trainclasses))
    errorcount = 0
    for docindex in testset:
        wordvector = setofwords2vec(vocablist,doclist[docindex])
        if classifynb(array(wordvector),p0v,p1v,pspam) != classlist[docindex]:
           errorcount += 1
    print ('the error rate is:',float(errorcount)/len(testset))
   
   
   