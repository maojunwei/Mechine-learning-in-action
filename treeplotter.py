# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 16:41:16 2017
决策树注解
@author: 001
"""
import matplotlib.pyplot as plt
decisionnode = dict(boxstyle = "sawtooth", fc = "0.8")
leafnode = dict(boxstyle = "round4",fc = "0.8")
arrow_args = dict(arrowstyle="<-")

def plotnode(nodetxt,centerpt,parentpt,nodetype):  #使用文本注解绘制树节点
    createplot.ax1.annotate(nodetxt,xy = parentpt,xycoords = 'axes fraction',xytext = centerpt,\
                            textcoords = 'axes fraction',va = 'center',bbox = nodetype,arrowprops = arrow_args)

def createplot():
    fig = plt.figure(1,facecolor = 'white')
    fig.clf()
    createplot.ax1 = plt.subplot(111,frameon=False)
    plotnode(U'decision node',(0.5,0.1),(0.1,0.5),decisionnode)
    plotnode(U'leaf node',(0.8,0.1),(0.3,0.8),leafnode)
    plt.show()
    
def getnumleafs(mytree):
    numleafs = 0
    firstsides = list(mytree.keys())
    firststr = firstsides[0]
    seconddict = mytree[firststr]
    for key in seconddict.keys():
        if type(seconddict[key])== dict:
            numleafs += getnumleafs(seconddict[key])
        else: numleafs += 1
    return numleafs

def gettreedepth(mytree):
    maxdepth = 0
    firstsides = list(mytree.keys())
    firststr = firstsides[0]
    seconddict = mytree[firststr]
    for key in seconddict.keys():
         if type(seconddict[key])== dict:
            thisdepth = 1 + gettreedepth(seconddict[key])
         else: thisdepth = 1
         if thisdepth > maxdepth: maxdepth = thisdepth
    return maxdepth

def retrievetree(i):
    listoftrees = [{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}},
                   {'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}]
    return listoftrees[i]

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
    
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getnumleafs(myTree)  
    depth = gettreedepth(myTree)
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]#找到输入的第一个元素
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':   
            plotTree(secondDict[key],cntrPt,str(key))        
        else:   
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    
    plotTree.totalW = float(getnumleafs(inTree))
    plotTree.totalD = float(gettreedepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()
    

   