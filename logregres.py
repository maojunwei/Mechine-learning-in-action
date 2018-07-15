# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 21:32:12 2017

@author: 001
"""
from numpy import*
def loaddataset():
    datamat = [];  labelmat = []
    fr = open('testset.txt')
    for line in fr.readlines():
        linearr = line.strip().split()
        datamat.append([1.0,float(linearr[0]),float(linearr[1])])
        labelmat.append(int(linearr[2]))
    return datamat,labelmat

def sigmoid(inx):
    return 1.0/(1 + exp(-inx))

#logistic回归梯度上升优化算法
def gradascent(datamatin,classlabels):
    datamatrix = mat(datamatin)
    labelmat = mat(classlabels).transpose()
    m,n = shape(datamatrix)
    alpha = 0.001
    maxcycles = 500
    weights = ones((n,1))
    for k in range(maxcycles):
        h = sigmoid(datamatrix * weights)
        error = (labelmat - h)
        weights = weights + alpha * datamatrix.transpose() * error
    return weights

#画出数据集和logistic回归最佳拟合直线的函数
def plotbestfit(weights):
    import matplotlib.pyplot as plt
    datmat,labelmat = loaddataset()
    dataarr = array(datmat)
    n = shape(dataarr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelmat[i]) == 1:
            xcord1.append(dataarr[i,1]); ycord1.append(dataarr[i,2])
        else:
            xcord2.append(dataarr[i,1]); ycord2.append(dataarr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c = 'red',marker = 's')
    ax.scatter(xcord2,ycord2,s=30,c= 'green')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0] - weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
#随机梯度上升算法    
def stocgradascent0(datamatrix,classlabels):
    m,n = shape(datamatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(datamatrix[i] * weights))
        error = classlabels[i] - h
        weights = weights + alpha * error * datamatrix[i]
    return weights

#改进的随机梯度上升算法（保证迭代次数）
def stocgradascent1(datamatrix,classlabels,numiter=150):
    m,n = shape(datamatrix)
    weights = ones(n)
    for j in range(numiter):     
        dataindex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01
            randindex = int(random.uniform(0,len(dataindex)))
            h = sigmoid(sum(datamatrix[randindex] * weights))
            error = classlabels[randindex] - h
            weights = weights + alpha * error * datamatrix[randindex]
            del(dataindex[randindex])
    return weights

#logistic回归分类函数
def classifyvector(inx,weights):
    prob = sigmoid(sum(inx*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colictest():
    frtrain = open('horseColicTraining.txt')
    frtest = open('horseColicTest.txt')
    trainingset = []; traininglabels = []
    for line in frtrain.readlines():
        currline = line.strip().split('\t')
        linearr = []
        for i in range(21):
            linearr.append(float(currline[i]))
        trainingset.append(linearr)
        traininglabels.append(float(currline[21]))
    trainweights = stocgradascent1(array(trainingset),traininglabels,500)
    errorcount = 0; numtestvec = 0.0
    for line in frtest.readlines():
        numtestvec += 1.0
        currline = line.strip().split('\t')
        linearr = []
        for i in range(21):
            linearr.append(float(currline[i]))
        if int(classifyvector(array(linearr),trainweights)) != int(currline[21]):
            errorcount += 1
    errorrate = (float(errorcount)/numtestvec)
    print("the error rate of this test is: %f" % errorrate)
    return errorrate

def multitest():
    numtests = 10;   errorsum = 0.0
    for k in range(numtests):
        errorsum += colictest()
    print ("after %d iterations the average error rate is: %f" %(numtests,errorsum/float(numtests)))