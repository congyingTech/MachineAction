import numpy as np
'''
加载Data,有两个特征值x1，和x2，位于前两列，并且添加x0=1列，第三列是label标签，
'''
def loadDataSet():
    dataMat = []; labelMat = []
    with open('testSet.txt') as fr:
        for line in fr.readlines():
            lineArr = line.split('\t')
            # print(lineArr)
            dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

'''
logistics函数
'''
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

'''
梯度上升法求权重w，每次用全部的样本求下一步的w
'''
def gradientAsc(dataMat, labelMat):
    a = 0.001 #梯度上升的步长
    w = np.ones([3,1])
    maxCycles = 500
    dataMatrix = np.mat(dataMat)
    labelMatrix = np.mat(labelMat)
    # print(np.shape(dataMatrix),np.shape(w))
    #print(np.dtype(w))
    # print(dataMatrix)
    # print(dataMatrix.dot(w))
    for i in range(maxCycles):
        #print(dataMatrix*w)
        h = sigmoid(dataMatrix*w)
        #print(type(h))
        error = labelMatrix.transpose() - h
        w = w + a*dataMatrix.transpose()*error #更新w
    return w

'''
随机梯度上升——每次用一个样本求w
'''
def stocGradientAsc(dataMat, labelMat):
    m,n = np.shape(dataMat)
    a = 0.01
    w = np.ones(n)
    for i in range(m):
        h = sigmoid(np.sum(dataMat[i]*w)) #h是一个数值
        error = labelMat[i] - h #error是一个数值
        w = w + a*error*np.array(dataMat[i])
    return w
'''
改进的随机梯度,增加150轮的次数计算w
'''
def stocGradientAsc1(dataMat, labelMat, numIter=150):
    m, n = np.shape(dataMat)
    w = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            a = 4.0/(1+j+i)+0.01 #a在每次迭代的时候都进行调整
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(np.sum(w*dataMat[randIndex]))
            error = labelMat[randIndex] - h
            w = w + a*error*np.array(dataMat[randIndex])
            del(dataIndex[randIndex])
    return w



'''
根据求的w绘制决策边界
'''
def plotBestFit(w):
    import matplotlib.pyplot as plt
    if not isinstance(w,np.ndarray):
        weights = w.getA() #将w从矩阵的形式转换为array的形式
    weights = w
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i][1]); ycord1.append(dataArr[i][2])
        else:
            xcord2.append(dataArr[i][1]); ycord2.append(dataArr[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #y=1的点的坐标为(xcord1, ycord1), y=0的点的坐标为(xcord2, ycord2)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1]*x) / weights[2] #绘制拟合直线
    ax.plot(x, y)
    plt.xlabel('X1');plt.ylabel('X2') #给横纵坐标指定标签
    plt.show()


def classifyVector(inX, w):
    prob = sigmoid(np.sum(inX*w))
    if prob>0.5:
        return 1
    else:
        return 0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21): #有21个特征值
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    w = stocGradientAsc1(trainingSet, trainingLabels)
    errCount = 0; numTest = 0
    for line in frTest.readlines():
        numTest += 1
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        preRes = classifyVector(lineArr, w)
        if preRes != float(currLine[21]):
            errCount += 1
    print('error rate = ', errCount/numTest)
    return errCount/numTest

def mutiTest():
    numTests = 10; errorSum = 0
    for k in range(numTests):
        errorSum += colicTest()
    print('average error rate is ', errorSum/numTests)




if __name__ == "__main__":
    dataMat, labelMat = loadDataSet()
    #w = gradientAsc(dataMat, labelMat)
    #w1 = stocGradientAsc(dataMat, labelMat)
    #w2 = stocGradientAsc1(dataMat, labelMat)

    #plotBestFit(w)
    #plotBestFit(w1)
    #plotBestFit(w2)
    mutiTest()