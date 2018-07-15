# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 21:01:29 2017

@author: 001
"""
from math import log
import operator
import pickle

def calcshannoent(dataset):  #计算给定数据集的香农熵
    numentries = len(dataset)
    labelcounts = {}
    for featurevec in dataset:
        currentlabel = featurevec[-1]
        if currentlabel not in labelcounts.keys():
            labelcounts[currentlabel] = 0
        labelcounts[currentlabel] += 1
    shannonent = 0.0
    for key in labelcounts:
        prob = float(labelcounts[key])/numentries
        shannonent -= prob * log(prob,2)
    return shannonent         

def createdataset():   #创建示例特征集
    dataset = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataset,labels

def splitdataset(dataset,axis,value):  #按照所给特征划分数据集
    retdataset = []
    for featvec in dataset:
        if featvec[axis] == value:
            reducefeatvec = featvec[:axis]
            reducefeatvec.extend(featvec[axis+1:])
            retdataset.append(reducefeatvec)
    return retdataset

def choosebestfeaturetosplit(dataset):  #选择最好的的数据划分方式
    numfeatures = len(dataset[0]) - 1
    baseentropy = calcshannoent(dataset)
    bestinfogain = 0.0;   bestfeature = -1
    for i in range(numfeatures):
        featlist = [example[i] for example in dataset] #第i特征项遍历
        uniquevals = set(featlist)
        newentropy = 0.0
        for value in uniquevals:
            subdataset = splitdataset(dataset,i,value)
            prob = len(subdataset)/float(len(dataset))
            newentropy += prob * calcshannoent(subdataset)
        infogain = baseentropy - newentropy
        if (infogain > bestinfogain):
            bestinfogain = infogain
            bestfeature = i
    return bestfeature

def majoritycnt(classlist):  #当决策树遍历完数据集的特征属性，分支下仍有不同分类时，选择次数最多的分类,(只剩标签项)
    classcount = {}
    for vote in classlist:
        if vote not in classcount.keys():  classcount[vote] = 0 #键值不存在，则扩展字典键
        classcount[vote] += 1
    sortedclasscount = sorted(classcount.items(),key = operator.itemgetter(1),reverse = True)
    return sortedclasscount[0][0]

def createtree(dataset,labels):  #(特征属性)创建决策树
    classlist = [example[-1] for example in dataset]
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    if len(dataset[0]) == 1:
        return majoritycnt(dataset)
    bestfeat = choosebestfeaturetosplit(dataset)
    bestfeatlabel = labels[bestfeat]
    mytree = {bestfeatlabel:{}}
    del(labels[bestfeat])
    featvalues = [example[bestfeat] for example in dataset] 
    uniquevals = set(featvalues)
    for value in uniquevals:
        sublabels = labels[:]
        mytree[bestfeatlabel][value] = createtree(splitdataset(dataset,bestfeat,value),sublabels)
    return mytree

### 使用决策树执行分类
def classify(inputtree,featlabels,testvec):
    firstside = list(inputtree.keys())
    firststr = firstside[0]
    seconddict = inputtree[firststr]
    featureindex = featlabels.index(firststr)
    for i in seconddict.keys():
        print(i)
    for key in seconddict.keys():
        if testvec[featureindex] == key:
            if type(seconddict[key]) == dict:
                classlabel = classify(seconddict[key],featlabels,testvec)
            else: classlabel = seconddict[key]
    return classlabel
    
### 使用pickle模块存储决策树
def storetree(inputtree,filename):  #写入
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputtree,fw)
    fw.close()
    
def grabtree(filename):         #读出
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

#使用决策树预测隐形眼镜类型
'''
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lenseslabels = ['age','prescript','astigmatic','tearrate']
lensestree = decisiontree.createtree(lenses,lenselabels)
treeplotter.createPlot(lensestree)
    
    

