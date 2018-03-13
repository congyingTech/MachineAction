from math import log
import operator

'''
计算熵
'''
def calShannonEnt(dataSet):
    n = len(dataSet)
    labelDict = {}
    for item in dataSet:
        if item[-1] not in labelDict:
            labelDict[item[-1]] = 0
        labelDict[item[-1]] += 1
    shannonEnt = 0
    for i in labelDict:
        p = labelDict[i] / n
        shannonEnt -= p*log(p, 2)
    return shannonEnt
'''
造数据
'''
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

'''
按照给定特征划分数据集
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            #reducedFeatVec的取值区间是 0≤x<axis和 axis+1≤x<len,所以被选定的特征没有在其中
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    #print('切分的data——', retDataSet)
    return retDataSet
'''
选择最好的数据集划分方式
'''
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1 #特征值的数量
    baseEntropy = calShannonEnt(dataSet)
    bestInfoGain = 0; bestFeature = -1  #初始化最佳的信息增益和最佳的特征选择
    for i in range(numFeatures): #分别对i个特征分割data，计算其信息增益，找出信息增益infoGain最大的那个i
        featList = [example[i] for example in dataSet] #找出feature i的那一列的值
        uniqueVals = set(featList) #不重复的value值，代表着feature-i有几种元素
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i , value) #对第i个特征值的第value个类别划分
            #print('value %s 切分的子集%s' %(value, subDataSet) )
            p = len(subDataSet)/len(dataSet)
            newEntropy += p*calShannonEnt(subDataSet) #newEntropy是对各个value的熵的期望
        infoGain = baseEntropy - newEntropy #feature-i的信息增益
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

'''
投票表决法
'''
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)
    #print(sortedClassCount)
    return sortedClassCount[0][0]

'''
创建树
'''
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0])  == len(classList): #如果classList中只有一个类别，表示分类完毕
        return classList[0]
    if len(dataSet[0]) == 1: #如果使用完了所有的特征，仍然不能将数据集划分为仅包含唯一类别的分组，用投票表决法
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet) #根结点
    #print(bestFeat)
    bestFeatLabel = labels[bestFeat] #'no surfacing', 'flippers'中的某个
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat]) #删除最佳特征后剩余的特征，再从其中找最佳
    featValues = [example[bestFeat] for example in dataSet] #最佳的特征所在列的值
    uniqueVals = set(featValues) #所在列的不同的值
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree



def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr) #找到根结点在特征label中的位置,也就是根结点是第几个特征，从而对应到testVec的值，0或者1，也就是走向信息，从而决定往哪个方向走
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)


'''
========================预测患者佩戴隐形眼睛类型====================
'''
def getData(filename):
    with open(filename) as fr:
        lenses =  [line.strip().split('\t') for line in fr.readlines()]
        labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    return lenses,labels

def main():
    lenses, labels = getData('lenses.txt')
    mytree = createTree(lenses,labels)
    lenses, labels = getData('lenses.txt')
    print(classify(mytree, labels, ['pre','hyper','no','normal']))



if __name__ == "__main__":
    # myData, labels = createDataSet()
    # mytree = createTree(myData, labels)
    # myData, labels = createDataSet()
    # classify(mytree, labels,[1,0])
    main()




