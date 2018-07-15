# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 19:31:26 2017

@author: 001
"""
from numpy import *
def loaddataset(filename,delim = '\t'):
    fr = open(filename)
    stringarr = [line.strip().split(delim) for line in fr.readlines()]
    datarr = [list(map(float,line)) for line in stringarr]
    return mat(datarr)

def pca(datamat,topnfeat=9999999):
    meanvals = mean(datamat,axis=0)
    meanremoved = datamat - meanvals
    covmat = cov(meanremoved,rowvar=0)
    eigvals,eigvects = linalg.eig(mat(covmat))
    eigvalind = argsort(eigvals)
    eigvalind = eigvalind[:-(topnfeat+1):-1]
    redeigvects = eigvects[:,eigvalind]
    lowddatamat = meanremoved * redeigvects
    reconmat = (lowddatamat * redeigvects.T) + meanvals
    return lowddatamat,reconmat

def replacenanwithmean():
    datmat = loaddataset('secom.data','')
    numfeat = shape(datmat)[1]
    for i in range(numfeat):
        meanval = mean(datmat[nonzero(~isnan(datmat[:,i].A))[0],i])
        datmat[nonzero(isnan(datmat[:,i].A))[0],i] = meanval
            
    return datmat

#降维后数据与原始数据图示
def figpca(olddata,newdata):
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.scatter(olddata[:,0],flatten().A[0],olddata[:,1].flatten().A[0],marker='^',s=90)
    ax.scatter(olddata[:,0],olddata[:,1],marker='^',s=90)
    #ax.scatter(newdata[:,0],flatten().A[0],newdata[:,1].flatten().A[0],marker='o',s=50,c='red')
    ax.scatter(newdata[:,0],newdata[:,1],marker='o',s=50,c='red')
    plt.show()
    
#将nan替换成平均值，处理缺失值
def replacenanwithmean():
    datmat = loaddataset('secom.data',' ')
    numfeat = shape(datmat)[1]
    for i in range(numfeat):
        meanval = mean(datamat[nonzero(~isnan(datamat[:,i].A))[0],i])
        datmat[nonzero(isnan(datmat[:,i].A))[0],i] = meanval
    return datmat

#确定降维维数
def percentage2n(val,percentage):
    sortarry = sort(val)
    sortarray = sortarry[-1::-1]#逆序
    arraysum = sum(sortarray)
    tmpsum = 0
    num = 0
    for i in sortarray:
        tmpsum+=i
        num +=1
        if tmpsum >= arraysum*percentage:
           return num  #合理维数
        
#改进pca,百分比版本
def pca1(datamat,percentage = 0.99):
    meanvals = mean(datamat,axis=0)
    meanremoved = datamat - meanvals
    covmat = cov(meanremoved,rowvar=0)
    eigvals,eigvects = linalg.eig(mat(covmat))
    eigvalind = argsort(eigvals)
    jiangw = percentage2n(eigvals,percentage)
    eigvalind = eigvalind[:-(jiangw+1):-1]
    redeigvects = eigvects[:,eigvalind]
    lowddatamat = meanremoved * redeigvects
    reconmat = (lowddatamat * redeigvects.T) + meanvals
    return lowddatamat,reconmat
    
    
    
    