#-*- coding: utf-8 -*-

import numpy as np
import operator
import matplotlib
from os import listdir
from PIL import Image

def createDataset():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

'''
inX-用于分类的输入向量inX
dataSet-输入的训练样本
'''
def classify0(inX, dataSet,labels, k):

    dataSetSize = dataSet.shape[0] #有几行数据，也就是点的个数
    testMat = np.tile(inX, (dataSetSize,1))
    #print(testMat)
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet #tile是repeat的意思
    sqDiffMat = diffMat ** 2
    distance = sqDiffMat.sum(axis=1) #而当加入axis=1以后就是将一个矩阵的每一行向量相加

    sortedDistIndicies = distance.argsort() #我们发现argsort()函数是将distance中的元素从小到大排列，提取其对应的index(索引)，然后输出到sortedDistIndicies
    classCount={} #初始化一个字典
    for i in range(k):
        #下一步是得到按distance从小到大序排的label，例如A，B，A，B
        voteIlabel = labels[sortedDistIndicies[i]] #i=0时，读取distance最小的那个的index，对应读取其label
        #下一步是对已经存在于classCount中的voteIlabel进行累加
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1 #dict.get(key, default=None)  key -- 字典中要查找的键。default -- 如果指定键的值不存在时，返回该默认值值。
    #返回一个迭代器，通过itemgetter选定进行排序的key，这里应该是字典中负责计数的count，方便后面计算label出现的频率
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse=True) #iteritems返回迭代器
    #print(type(sortedClassCount))#sortedClassCount是一个list，所以要取[0][0]表示最大的那个类别
    return sortedClassCount[0][0]





'''
=================================网站约会系统================================
'''

'''读取文件数据转换为矩阵'''
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip() #删除头尾的空格字符串
        listFromLine = line.split('\t')
        #对returnMat第index行赋值，将其转换为矩阵形式
        returnMat[index, :] = listFromLine[0:3]
        #最后一列数据为结果集
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat, classLabelVector

'''
归一化特征值
'''
def autoNorm(dataSet):
    minVals = dataSet.min(0) #axis=0; 每列的最小值   min(1)axis=1；每行的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals #一行n列的矩阵
    normDataSet = np.zeros(np.shape(dataSet)) #用来存储归一化的dataSet
    m = dataSet.shape[0] #这是矩阵的行数
    normDataSet = dataSet - np.tile(minVals,(m,1)) #tile相当于将一行n列的矩阵复制成了m行
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet, ranges, minVals


'''
测试错误率
'''
def datingClassTest():
    hoRatio = 0.10 #0.1比例的数据是用于测试的
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio) #用于测试的数据的数量
    errorCount = 0.0
    #normMat的前0.1行作为测试数据，后0.9行作为训练数据进行预测。normMat[numTestVecs:m, :]
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m],3)
        print("the predict answer is %d , the real answer is %d" % (classifierResult,datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1
    print('error rate is :%f' % (errorCount/numTestVecs))



'''
通过算法对输入的数据进行预测
'''
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']

    percentTats = float(input("video games time?"))
    ffMiles = float(input("flier miles per year?"))
    iceCream = float(input("ice cream per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels,3)
    print('You will probably like this person:', resultList[classifierResult-1])




'''
==================================手写识别系统====================================
'''

'''
将图像转换为向量
'''
def img2vector(filename):
    returnVect = np.zeros([1,1024])
    fr = open(filename)
    for i in range(32):#按行读取fr
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect



def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('CH01kNN/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros([m,1024])
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('CH01kNN/trainingDigits/%s'%fileNameStr)

    testFileList = listdir('CH01kNN/testDigits')
    errorCount = 0
    n = len(testFileList)
    for i in range(n):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        testVector = img2vector('CH01kNN/testDigits/%s'%fileNameStr)
        predictResult = classify0(testVector, trainingMat,hwLabels,3)
        print('the predict answer is %d'%predictResult, 'the real answer is %d'%classNumStr)
        if predictResult != classNumStr:
            errorCount += 1
    #print('total error is % d'% errorCount)
    #print('error rate is %f' % (errorCount/n))
    return trainingMat,hwLabels

'''
图片转换为矩阵
'''
def ImageToMat(filename):
    im = Image.open(filename)
    im = im.resize((32,32))
    width, height = im.size
    #im.show()
    im = im.convert('L') #将图片转换为黑白
    with open('CH01kNN/test.txt', 'wt') as f:
        for i in range(height):
            for j in range(width):
                pixel = int(im.getpixel((j,i))/255)

                if pixel==1:
                    pixel = 0
                elif pixel==0:
                    pixel = 1

                f.write(str(pixel))
                if j == width-1:
                    f.write('\n')
        f.close()
'''
通过算法对转换成功的图片进行预测
'''
def ClassifyHandwriting():
    filename = input('where is file?')
    newMat = img2vector(filename) #将32*32的数字图像转换为矩阵
    trainingMat, labels = handwritingClassTest()
    result = classify0(newMat,trainingMat,labels,3)
    print(result)





if __name__ == "__main__":
    group, labels = createDataset()
    #print(classify0([0,1.3], group, labels,3))
    datingDataMat, dating = file2matrix('CH01kNN/datingTestSet2.txt')
    #print(datingDataMat)
   # print(dating)
    print(handwritingClassTest)
