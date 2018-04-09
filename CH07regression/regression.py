from numpy import *
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0: #判断是否为奇异矩阵
        print('This matrix is singular, cannot be inverse')
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws


def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))#创建对角矩阵
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * (weights*yMat))
    return testPoint*ws

def lwlrTest(testArr, xArr, yArr, k=1):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr,k)
    return yHat

def lwlrMain():
    xArr, yArr = loadDataSet('ex0.txt')
    yHat = lwlrTest(xArr, xArr,yArr,0.003)
    xMat = mat(xArr)
    # print('xMat',len(xMat[:,1]))
    # print(shape(xMat[:,1]))
    srtInd = xMat[:,1].argsort(0)
    # print('srtInd', srtInd)
    xSort = xMat[srtInd][:,0,:]
    #print(xSort[:,1])
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1], yHat[srtInd])
    print(xMat[:, 1].flatten().A[0])
    ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red') #.A return ndarray
    plt.show()

def main():
    xArr, yArr = loadDataSet('ex0.txt')
    ws = standRegres(xArr, yArr)
    print(ws)
    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat*ws
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    #求解相关系数计算预测值与真实值的相关性，对角线的是yMat和自己的相关性，斜对角是与yHat的相关性
    print('相关系数', corrcoef(yHat.T, yMat))
    ax.plot(xCopy[:,1], yHat)

    plt.show()

def rssError(yArr, yHatArr):
    return ((yArr-yHatArr)**2).sum()

def abaloneMain():
    xArr, yArr = loadDataSet('abalone.txt')
    yHat01 = lwlrTest(xArr[0:99], xArr[0:99], yArr[0:99], 0.1)
    yHat1 = lwlrTest(xArr[0:99], xArr[0:99], yArr[0:99], 1)
    yHat10 = lwlrTest(xArr[0:99], xArr[0:99], yArr[0:99], 10)


    print('0.1 error',rssError(yArr[0:99], yHat01.T))
    print('1 error',rssError(yArr[0:99], yHat1.T))
    print('10 error',rssError(yArr[0:99], yHat10.T))


def ridgeRegression(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = denom.I * (xMat.T * yMat)
    return ws
def ridgeTest(xArr, yArr):
    pass



if __name__ == '__main__':
    #main()
    #lwlrMain()
    abaloneMain()
