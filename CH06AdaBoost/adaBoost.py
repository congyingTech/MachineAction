from numpy import *


def loadSimpData():
    dataMat = matrix([[1, 2.1],
                    [2, 1.1],
                    [1.3, 1],
                    [1, 1],
                    [2,1]]
                    )
    classLabels = [1,1,-1, -1, 1]
    return dataMat, classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones([shape(dataMatrix)[0], 1])
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1
    return retArray


def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10; bestStump = {}; bestClassEst = mat(zeros([m,1]))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min(); rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax-rangeMin)/numSteps #步长的大小
        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones([m,1]))
                errArr[predictedVals==labelMat] = 0
                weightedError = D.T * errArr
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones([m, 1])/m)
    aggClassEst = mat(zeros([m, 1]))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print('D is', D.T)
        #alpha是放缩的大小,是一个正数，为了避免被除数为0，设置一个不为0的很小的值
        alpha = float(0.5 * log((1-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print('classEst', classEst.T)
        expon = multiply(-1*alpha*mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst#通过alpha加强后的classEst，即每个数据点的类别估计累计值
        print('aggClassEst',aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones([m,1]))
        errorRate = aggErrors.sum()/m
        print('total error', errorRate, '\n')
        if errorRate == 0.0:break
    return weakClassArr, aggClassEst

def adaClassify(dataToClass, classifierArr):
    dataMatrix = mat(dataToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros([m, 1]))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq']
                                 )
        aggClassEst += classifierArr[i]['alpha']*classEst
    return sign(aggClassEst)

#自适应数据加载函数
def loadDataSet(fileName):
    #特征的数量
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat=[]; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat
def main():
    dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, labelArr, 50)
    testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    predictResult = adaClassify(testArr, weakClassArr)
    errArr = mat(ones([len(testArr),1]))
    errorCount = errArr[predictResult!=mat(testLabelArr).T].sum()
    print(errorCount)
    print('error rate is', errorCount/len(testArr))
    plotROC(aggClassEst.T, labelArr)

def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #面积为1的矩形的右上角,表示绘制光标的位置——浮点数二元组
    ySum = 0 #用于计算AUC的值
    numPosClass = sum(array(classLabels)==1)#分类为正例的个数
    yStep = 1/float(numPosClass)  #这是y轴的刻度
    xStep = 1/float(len(classLabels) - numPosClass) #这是x轴的刻度，len(classLabels) - numPosClass为负例的个数
    sortedIndices = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndices.tolist()[0]:
        if classLabels[index] == 1.0:#如果是1的话，那就是正例，求真正例
            delX = 0; delY=yStep;
        else: #如果是-1的话，那就是
            delX = xStep; delY=0;
            ySum += cur[1]
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate') #x轴是假阳率
    plt.ylabel('True Positive Rate') #y轴是真阳率
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print('AUC is:', ySum*xStep)







if __name__ == "__main__":
    # D = mat(ones([5,1])/5)
    # dataMat, classLabels = loadSimpData()
    # bestStump, minError, bestClassEst = buildStump(dataMat, classLabels, D)
    # print(bestStump, minError, bestClassEst)
    # classifierArray = adaBoostTrainDS(dataMat, classLabels,9)
    # print(classifierArray)
    # print(adaClassify(dataMat, classifierArray))
    # print(adaClassify([0,1],classifierArray))
    main()

