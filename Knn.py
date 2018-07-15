# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:04:15 2017

@author: 毛俊伟
"""
'''
K近邻算法，书上的例程
'''
from numpy import *
import operator     #python中内置的操作符函数接口
from os import listdir  #导入函数listdir

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','B','B','B']
    return group,labels

def classify0(inX, dataSet, labels, k):      #k-近邻算法
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()  #返回数组值由大到小的索引值
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key= operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):    #从文本文件中解析数据
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOflines = len(arrayOlines)
    returnMat = zeros((numberOflines,3))
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataset):   #归一化数值
    minvals = dataset.min(0)
    maxvals = dataset.max(0)
    ranges = maxvals - minvals
    normdataset = zeros(shape(dataset))
    m = dataset.shape[0]
    normdataset = dataset - tile(minvals,(m,1))
    normdataset = normdataset/tile(ranges,(m,1))
    return normdataset,ranges,minvals

def datingclasstest(): #约会网站测试
    horatio = 0.10 
    datingDataMat,datinglabels = file2matrix('datingTestSet2.txt')
    normmat,ranges,minvals = autoNorm(datingDataMat)
    m = normmat.shape[0]
    numtestvecs = int(m*horatio)  #百分之十的数据用作测试
    errorcount = 0.0
    for i in range(numtestvecs):
        classifierResult = classify0(normmat[i,:],normmat[numtestvecs:m,:],datinglabels[numtestvecs:],3)
        print ("the classifier came back with: %d, the real answer is: %d" %(classifierResult,datinglabels[i]))
        if (classifierResult != datinglabels[i]): errorcount += 1.0
    print ("the total error rate is: %f" %(errorcount/float(numtestvecs)))
    
def classifyperson():   #完整可用系统
    resultlist = ['not at all','in small doses','in large doses']
    percentats = float(input("percentage of time spent playing vedio games?"))
    ffmiles = float(input("frequent flier miles earned per year?"))
    icecream = float(input("liters of ice cream consumed per year"))
    datingdatamat,datinglabels = file2matrix('datingTestSet2.txt')
    normmat,ranges,minvals = autoNorm(datingdatamat)
    inarr = array([ffmiles,percentats,icecream])
    classifierresult = classify0((inarr-minvals)/ranges,normmat,datinglabels,3)
    print ("You will probably like this this person:",resultlist[classifierresult - 1])
    
def fig():   #散点图
    import matplotlib
    import matplotlib.pyplot as plt
    dataset,labels = file2matrix('datingTestSet2.txt')
    fig = plt.figure()
    plt.ylabel('Y axis')
    plt.xlabel('X axis')
    #plt.legend(loc = 'first and second dimension')
    ax = fig.add_subplot(111)
    ax.scatter(dataset[:,1],dataset[:,2],15.0*array(labels), 15.0*array(labels))
    plt.show()
    
def img2vector(filename):    #读入图像向量
    returnvect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            returnvect[0,32*i+j] = int(linestr[j])
    return returnvect

def handwrtingclasstest():     #手写数字识别
    hwlabels = []
    trainingfilelist = listdir('trainingDigits')
    m = len(trainingfilelist)
    trainingmat = zeros((m,1024))
    for i in range(m):
        filenamestr = trainingfilelist[i]
        filestr = filenamestr.split('.')[0]
        classnumstr = int(filestr.split('_')[0])
        hwlabels.append(classnumstr)
        trainingmat[i,:] = img2vector('trainingDigits/%s' %filenamestr)
    testfilelist = listdir('testdigits')
    errorcount = 0.0
    mtest = len(testfilelist)
    for i in range(mtest):
        filenamestr = testfilelist[i]
        filestr = filenamestr.split('.')[0]
        classnumstr = int(filestr.split('_')[0])
        vectorundertest = img2vector('testdigits/%s' % filenamestr)
        classifierresult = classify0(vectorundertest,trainingmat,hwlabels,3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierresult, classnumstr))
        if (classifierresult != classnumstr): errorcount += 1.0
    print ("\n the total number of errors is: %d" % errorcount)
    print ("\n the total error rate is: %f" % (errorcount/float(mtest)))
        
    