from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat

'''相似度计算——欧式距离'''
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA-vecB, 2)))

'''随机选取k个簇心，簇心的计算是min + rangeJ*random(k,1)'''
#随机生成min到max之间的值
#其中rand(k,1)是高斯分布随机数，k行1列
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros([k, n])) #k个簇心是k行n列的
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:, j] = minJ + rangeJ* random.rand(k, 1)
    return centroids

