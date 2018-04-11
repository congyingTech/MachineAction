
#因为只对特征进行二元切分，所以可以固定树的数据结构：
#包括：待切分特征，待切分特征值，右子树，左子树
class treeNode():
    def __init__(self, feat, val, right, left):
        featureToSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left

from numpy import *
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine)) #将每行的数据映射成浮点数
        dataMat.append(fltLine)
    return dataMat

#二元分割数据集dataSet是待分割数据集，feature是待分割特征，value是待分割特征值
def binSplitDataSet(dataSet, feature, value):
    #nonzero产生一个tuple，表示x的非零元素的索引，第一个array是非零元素的行号，第二个array是非零元素的列号
    #dataSet[:, feature] > value是指的选定的feature大于value的元素，返回True或者False
    #nonzero(dataSet[:, feature] > value)[0]是表示选定feature值大于value的元素的行索引
    #print(dataSet[:, feature])
    #print(dataSet[:, feature]>value)
    #print(dataSet)
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]#取大于某feature的value划分到左边
    #print(dataSet[:, feature] > value)
    #print(nonzero(dataSet[:, feature] <= value))
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1

#负责生成叶节点，回归树中就是不再进行切分时得到目标变量的均值
def regLeaf(dataSet):
    return mean(dataSet[:,-1])

#连续型数据计算混乱度——平方误差的总值（总方差）
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]


#leafType给出建立叶节点的函数; errType表示误差计算函数
def createTree(dataSet, leafType = regLeaf, errType=regErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:return val #此时是叶子节点
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet,leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree



def chooseBestSplit(dataSet, leafType =regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0] #容许的误差下降值
    tolN = ops[1] #切分的最少样本数
    #如果所有dataSet[:,-1]是结果值，相等，则退出
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):#对于每个特征
        #print(set(dataSet[:, featIndex].T.A[0]))
        for splitVal in set(dataSet[:, featIndex].T.A[0]):#对于每个特征值
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if(shape(mat0)[0]<tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if(S - bestS) < tolS:  #误差减少不多的情况下退出
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if(shape(mat0)[0]<tolN) or (shape(mat1)[0] < tolN): #切分数据集很小的情况下退出
        return None, leafType(dataSet)
    return bestIndex, bestValue

'''==========================后剪枝======================'''
def isTree(obj):
    return (type(obj).__name__=='dict')

#递归函数，从上到下直到找到两个叶节点为止，并计算平均值（又叫对树的塌陷处理）
def getMean(tree):
    if isTree(tree['right']): tree['right']=getMean(tree['right'])
    if isTree(tree['left']): tree['left']=getMean(tree['left'])
    return (tree['left']+tree['right'])/2

def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']): #直到不再是树
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        #print('tree-left.....',tree['left'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'], 2)) + sum(power(rSet[:-1]-tree['right'], 2))
        treeMean = (tree['left']+tree['right'])/2 #将左右两颗子树进行合并
        errorMerge = sum(power(testData[:-1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree
'''=======================模型树====================='''
def linearSolve(dataSet):
    m, n = shape(dataSet)
    X = mat(ones([m, n])); Y = mat(ones([m,1]))
    X[:,1:n] = dataSet[:, 0:n-1]; Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0:
        raise NameError('This matrix is singular, cannot inverse')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y
def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y-yHat, 2))

'''=========================应用========================='''


'''树回归的Eval'''
def regTreeEval(model, inDat):
    return float(model)

'''模型树的Eval'''
#inDat是待预测的特征值
def modelTreeEval(model, inDat):
    n = shape(inDat)[1] #特征的数量
    X = mat(ones([1, n+1])) #第一列是1
    X[:, 1:n+1] = inDat
    return float(X*model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)

def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros([m, 1]))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat








if __name__ == '__main__':
    # testMat = mat(eye(4))
    # print(testMat)
    # mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    # print(mat0, mat1)
    #dataMat = mat(loadDataSet('ex2.txt'))
    #print(dataMat)
    #print(dataMat)
    # tree = createTree(dataMat, ops=(0, 4))
    # print('origin tree ======================================',tree)
    # dataTest = mat(loadDataSet('ex2test.txt'))
    # prune(tree, dataTest)
    # print('prune tree ======================================',tree)
    #tree = createTree(dataMat, modelLeaf, modelErr, ops=(1,10))
    #print(tree)

    '''回归树'''
    trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    myTree = createTree(trainMat, ops=(1,20))
    print(myTree)
    yHat = createForeCast(myTree, testMat[:, 0])
    print(yHat)
    print(corrcoef(yHat, testMat[:,1], rowvar=0)[0,1])

    '''模型树'''
    myTree1 = createTree(trainMat, modelLeaf, modelErr, ops=(1,20))
    yHat1 = createForeCast(myTree1, testMat[:, 1], modelTreeEval)
    print(corrcoef(yHat1, testMat[:, 1], rowvar=0)[0,1])

