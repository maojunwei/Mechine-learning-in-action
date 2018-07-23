# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:45:35 2017
基于手写体数据进行线性SVM 验证实验
召回率、准确率和F1指标适用于二分类任务，多分类中逐一评估某个类别的这三个性能指标，把其他的类别作为负样本
对于多分类问题，svm实现的几种间接形式：
a.一对多法，某个类别的样本归为一类，其他剩余的样本归为另一类，k个类别的样本需要构建k个svm
b.一对一法，任意两个类别间设计一个SVM，因此k个类别需要(k*(k-1))/2个svm分类器（libsvmd多分类采用的方式）
c.层次支持向量机。层次分类法首先将手游类别分成两个子类，逐层划分
@author: mjw
"""
from sklearn.datasets import load_digits #sklearn内部的手写体数字图片数据集
digits = load_digits() #8*8像素矩阵图片，2D图片一维展开，1797*64（处理结构性信息的能力）
#digits.data.shape
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target,
                                                 test_size = 0.25,random_state = 33)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC #线性支持向量机，默认一对多
"""
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
C：目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0；  
loss ：指定损失函数  
penalty ：  
dual ：选择算法来解决对偶或原始优化问题。当n_samples > n_features 时dual=false。  
tol ：（default = 1e - 3）: svm结束标准的精度;  

multi_class：如果y输出类别包含多类，用来确定多类策略， ovr表示一对多，“crammer_singer”优化所有类别的一个共同的目标  
如果选择“crammer_singer”，损失、惩罚和优化将会被被忽略。 
 
fit_intercept ：  
intercept_scaling ：  
class_weight ：对于每一个类别i设置惩罚系数C = class_weight[i]*C,如果不给出，权重自动调整为 n_samples / (n_classes * np.bincount(y))  
verbose：跟多线程有关
"""
ss = StandardScaler()
X_train = ss.fit_transform(X_train )  #fit是为后续的API函数服务的
X_test = ss.transform(X_test)
lsvc = LinearSVC()
lsvc.fit(X_train,y_train)
y_predict = lsvc.predict(X_test)
print ('Accuracy of Linear SVC is:',lsvc.score(X_test,y_test))
from sklearn.metrics import classification_report
print (classification_report(y_test,y_predict,target_names=digits.target_names.astype(str))) 

from sklearn.svm import SVC 
"""
#基于libsvm实现（一对一的方式），数据拟合的时间复杂度是数据样本的二次方
SVC参数解释 
（1）C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0； 
（2）kernel：参数选择有RBF, Linear, Poly, Sigmoid, 默认的是"RBF"; 
（3）degree：if you choose 'Poly' in param 2, this is effective, degree决定了多项式的最高次幂； 
（4）gamma：核函数的系数('Poly', 'RBF' and 'Sigmoid'), 默认是gamma = 1 / n_features; 
（5）coef0：核函数中的独立项，'RBF' and 'Poly'有效； 
（6）probablity: 可能性估计是否使用(true or false)； 
（7）shrinking：是否进行启发式； 
（8）tol（default = 1e - 3）: svm结束标准的精度; 
（9）cache_size: 制定训练所需要的内存（以MB为单位）； 
（10）class_weight: 每个类所占据的权重，不同的类设置不同的惩罚参数C, 缺省的话自适应； 
（11）verbose: 跟多线程有关，不大明白啥意思具体； 
（12）max_iter: 最大迭代次数，default = 1， if max_iter = -1, no limited; 

（13）decision_function_shape ： ‘ovo’ 一对一, ‘ovr’ 一对多  or None 无, default=None 
（14）random_state ：用于概率估计的数据重排时的伪随机数生成器的种子。 
 ps：7,8,9一般不考虑。
对于数据不平衡问题， 
"""
svc = SVC(decision_function_shape='ovo')
svc.fit(X_train,y_train)
y_predict1 = svc.predict(X_test)
print ('Accuracy of SVC is:',svc.score(X_test,y_test))
print (classification_report(y_test,y_predict1,target_names=digits.target_names.astype(str)))
#dec1 = svc.decision_function(X_test[0,:])    #返回的是样本距离超平面的距离  
#print ("SVC:",dec1)  