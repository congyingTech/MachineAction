import numpy as np
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

'''
SMO本是要确定最佳alpha对，简化版是便利每一个alpha，然后在剩下的alpha集合中随机选择另一个alpha
i是第一个alpha的下标，m是所有alpha的数目
只要函数值不等于输入值，函数就会进行随机选择
'''
def selectJrand(i, m):
    j = i
    while (j==i):
        j = int(np.random.uniform(0,m))
    return j

'''
H和L是修正过大或者过小的aj
'''
def clipAlpha(aj,H,L):
    if aj>H:
        aj = H
    if aj<L:
        aj = L
    return aj
'''
toler是容错率
'''
def smoSimple(dataMat, labelMat, C, toler, maxIter):
    dataMatrix = np.mat(dataMat);labelMatrix = np.mat(labelMat)
    b = 0; m,n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros([m,1]))
    iter = 0
    while (iter<maxIter):
        alphaPairsChanged = 0 #alpha是否改变的标志
        for i in range(m):
            #fXi是一个m行1列的预测结果
            fXi = alphas.transpose()*labelMatrix*dataMatrix*dataMatrix[i,:].T + b
            #Ei是误差,如果误差特别大，那么可以对å进行优化
            Ei = fXi - labelMatrix[i]
            #如果alphas很大的时候，会导致fXi变大，那么误差Ei也会很大，
            if((labelMatrix[i]*Ei < -toler and alphas[i]<C) or (labelMatrix[i]*Ei > toler and alphas[i]>0)):
                j = selectJrand(i,m) #这时候
                fXj = float()






if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('testSet.txt')
