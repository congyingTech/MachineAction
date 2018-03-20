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
    dataMatrix = np.mat(dataMat);labelMatrix = np.mat(labelMat).transpose() #m行1列
    b = 0; m,n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros([m,1]))
    iter = 0
    while (iter<maxIter):
        alphaPairsChanged = 0 #alpha是否改变的标志
        for i in range(m):
            #fXi是一个数值化的预测结果
            fXi = float(np.multiply(alphas,labelMatrix).T*(dataMatrix*dataMatrix[i,:].T))+b
            #Ei是误差,如果误差特别大，那么可以对å进行优化
            Ei = fXi- float(labelMatrix[i])
            #这一部分是违背KKT条件的：具体原理参考http://blog.csdn.net/on2way/article/details/47730367
            if((labelMatrix[i]*Ei < -toler and alphas[i]<C) or (labelMatrix[i]*Ei > toler and alphas[i]>0)):
                j = selectJrand(i,m)
                fXj = float(np.multiply(alphas, labelMatrix).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj-float(labelMatrix[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                #具体的推导参考http://blog.csdn.net/on2way/article/details/47730367
                #if y1 ≠ y2 : L = max(0, αold2−αold1), H = min(C, C + αold2−αold1)
                #if y1 = y2 : L = max(0, αold2 + αold1−C), H = min(C, αold2 + αold1)
                if(labelMatrix[i] != labelMatrix[j]):
                    L = max(0, alphas[j]-alphas[i])
                    H = min(C, C+alphas[j]-alphas[i])
                else:
                    L = max(0, alphas[i] + alphas[j]-C)
                    H = min(C, alphas[i]+alphas[j])
                if L==H:
                    print('L==H');continue
                eta = 2*dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - \
                      dataMatrix[j,:]*dataMatrix[j,:].T
                if eta>= 0:print('eta>=0');continue
                alphas[j] -= labelMatrix[j]*(Ei-Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold)<0.00001):
                    print('j not moving enough');continue
                alphas[i] += labelMatrix[j]*labelMatrix[i]*(alphaJold-alphas[j])

                b1 = b - Ei - labelMatrix[i]*(alphas[i]-alphaIold) * \
                     dataMatrix[i,:]*dataMatrix[i,:].T - \
                     labelMatrix[j]*(alphas[j] - alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMatrix[i]*(alphas[i]-alphaIold) * \
                     dataMatrix[i,:]*dataMatrix[j,:].T - \
                     labelMatrix[j]*(alphas[j] - alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T

                if (0<alphas[i]<C): b = b1 #如果åi调整好了，那么b就等于b1
                elif(0<alphas[j]<C): b = b2 #如果åj调整好了，那么b就等于b2
                else:
                    b = (b1+b2)/2.0
                alphaPairsChanged += 1
                print('iter %d pairs changed %d' % (iter, alphaPairsChanged))
        # 如果alphaPairs不再改变，那么进入下一轮的调整，否则证明alphas没有符合要求，需要继续进行调整。
        if(alphaPairsChanged == 0): iter+=1
        else:iter=0
        print("iter number=============%d"%iter)
    w = np.multiply(alphas, labelMatrix).T * dataMatrix
    return b, alphas, w

class optStruct: #创建一个数据结构
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros([self.m,1]))
        self.b = 0
        self.eCache = np.mat(np.zeros([self.m,2])) #第一位是表示有效位
        self.K = np.mat(np.zeros([self.m, self.m]))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

'''计算error'''
def calEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:,k]) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

'''已知i的情况下，启发式选择j'''
#启发式：第1个α选择后，其对应的点与实际标签有一个误差，属于边界之间α的所有点每个点都会有一个自己的误差，这个时候选择剩下的点与第一个α点产生误差之差最大的那个点。
def selectJ(i, oS, Ei):
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1, Ei]
    #type(eCache[:,0])
    #<class 'numpy.matrixlib.defmatrix.matrix'>
    #type(eCache[:,0].A)
    #<class 'numpy.ndarray'>
    #nonzero是把array中值非0的下标存在array[0],值为0的下标存在array[1]
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0] #validEcacheList记录不为0的行数值
    if(len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k==i:continue
            Ek = calEk(oS,k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE;Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calEk(oS, j)
    return j, Ej

'''更新符合条件的k和Ek，并将其加入eCache'''
def updateEk(oS, k):
    Ek = calEk(oS, k)
    oS.eCache[k] = [1, Ek]

'''用寻找到的åj，得到pairschange是否change的内循环'''
def innerL(i, oS):
    Ei = calEk(oS, i)
    #不符合KKT的条件的åi
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
            ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIOld = oS.alphas[i].copy()
        alphaJOld = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[i] + oS.alphas[j] - oS.C)
            H = min(oS.C, oS.alphas[i] + oS.alphas[j])
        if L == H:print('L==H');return 0
        eta = 2 * oS.K[i,j].T - oS.K[i,i] - \
              oS.K[j,j]
        if eta >= 0: print('eta>=0');return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        #更新符合条件的k进入eCache中，要求的条件是alpha在H和L之间
        updateEk(oS, j)

        if (abs(oS.alphas[j] - alphaJOld) < 0.00001):
            print('j not moving enough');return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJOld - oS.alphas[j])

        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaJOld) *oS.K[i,i] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJOld) * oS.K[i,j]

        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaJOld) * oS.K[i,j] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJOld) * oS.K[j,j]

        if (0 < oS.alphas[i] < oS.C):
            oS.b = b1  # 如果åi调整好了，那么b就等于b1
        elif (0 < oS.alphas[j] < oS.C):
            oS.b = b2  # 如果åj调整好了，那么b就等于b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

'''
外循环代码
'''
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin',0)):
    oS = optStruct(dataMatIn, classLabels, C, toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    #启发式的选择不是遍历所有的å，而是遍历在0-C之间(也就是在两条线之间的错误的点)错误的å
    #如果本轮部分遍历不再找到错误的å，那么进行全体的遍历，还可以找到新的错误的å，如此循环下去
    while(iter < maxIter) and ((alphaPairsChanged>0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet: #遍历所有的值
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            print('fullSet, iter: %d i:%d , pairs changed %d' %(iter, i, alphaPairsChanged))
            iter += 1
        else: #遍历非边界值,也就是alpha不等于0或者C的值，让两个条件相乘,求不为0的元素下标值
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print('non-bound, iter: %d i:%d , pairs changed %d' %(iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:entireSet=False
        elif(alphaPairsChanged==0): entireSet = True
        print('iteration number: %d' % iter)

    w = np.multiply(oS.alphas, oS.labelMat).T * oS.X
    return oS.b, oS.alphas, w

'''计算w的值'''
def calWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr);labelMat = np.mat(classLabels)
    m,n = np.shape(X)
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat, X[i,:].T)
    return w

'''kernel核函数'''
def kernelTrans(X, A, kTup):
    m,n = np.shape(X)
    K = np.mat(np.zeros([m,1]))
    if kTup[0] == 'lin': K = X*A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = np.exp(K/(-1*kTup[1]**2))
    else:
        raise NameError('Houston we have a problem..')
    return K

def testRbf(k1 = 1.3):
    dataMat, labelMat = loadDataSet('testSetRBF.txt')
    dataMatrix = np.mat(dataMat);labelMatrix = np.mat(labelMat).transpose()
    b, alphas, w = smoP(dataMatrix, labelMatrix, 200, 0.0001, 10000, ('rbf', k1))
    svIndex = np.nonzero(alphas.A > 0)[0]
    sVs = dataMatrix[svIndex]
    labelSvs = labelMatrix[svIndex]
    m,n = np.shape(dataMatrix)
    errCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMatrix[i,:],('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSvs, alphas[svIndex]) + b
        if np.sign(predict) != np.sign(labelMatrix[i]):
            errCount += 1
    errRate = errCount/m
    print('error rate is %f'% errRate)
if __name__ == '__main__':
    # dataMat, labelMat = loadDataSet('testSet.txt')
    # dataMatrix = np.mat(dataMat);labelMatrix = np.mat(labelMat).transpose() #m行1列
    # b, alphas, w = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    # print(b, w)
    # b, alphas, w = smoP(dataMatrix, labelMatrix, 0.6, 0.001, 40)
    # print(b,w)

    testRbf()
