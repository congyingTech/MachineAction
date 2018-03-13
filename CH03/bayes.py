import numpy as np
from math import log
import random
import feedparser
'''
创建一个包含在所有文档中出现的不重复的列表
'''
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 是侮辱性文字, 0 是正常言论，这六个结果对应上面的六条数据
    return postingList,classVec

'''
创建一个包含在所有文档中出现的不重复的列表
'''
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet|set(document)
    return list(vocabSet)

'''
将输入的inputSet对应到vocabList中的index，并在相应的index处赋值为1进行标记
——这样就将一篇文档转换成了词向量。
'''
#这是词集，每句话的词汇只能出现一次，不能有重复的情况
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word %s is not in my Vocabulary' % word)

    return returnVec

'''
词汇重复——文档词袋模式
计算每个词汇出现的数量
'''
def bagOfWords2VecMN(vocalbList, inputSet):
    returnVec = [0] * len(vocalbList)
    for word in inputSet:
        if word in vocalbList:
            returnVec[vocalbList.index(word)] += 1
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix) #参与训练的文档个数
    numWords = len(trainMatrix[0]) #全部词的个数
    pAbusive = sum(trainCategory) / numTrainDocs #侮辱性语言分类标签为1，侮辱类语言的比例p(c)
    #p0Num = np.zeros(numWords); p1Num = np.zeros(numWords) #p1Num是统计被认为是侮辱性语言的词出现的数量
    #因为p(wi|c1 or c0) = p(w0|c1)*p(w1|c1)*p(w2|c1)....所以一旦有一个是0，那么p(wi|c1)就为0，变得毫无意义，为了避免
    #这种情况
    p0Num = np.ones(numWords); p1Num=np.ones(numWords)
    #p0Denom = 0 ; p1Denom = 0
    p0Denom = 2 ; p1Denom = 2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1: #被标记为侮辱性语言
            p1Num += trainMatrix[i]  #p1Num是一行，numWords列的向量
            p1Denom += sum(trainMatrix[i]) #p1Denom是所有侮辱性语言出现的次数和，是一个具体的数字
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #为了避免连乘出现向下溢出的情况，解决办法是对乘积取自然对数
    #print('tesst',log(p1Num / p1Denom))
    #p1Vect = [log(i) for i in p1Num / p1Denom] #是1行，numWords列的向量，p(wi|c1),也就是在c=1条件下w的概率
    #p0Vect = [log(i) for i in p0Num / p0Denom] ##是1行，numWords列的向量，p(wi|c0)，也就是在c=0下w的概率
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def testNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as', classifyNB(thisDoc, p0V, p1V, pAb))

    testEntry = ['good', 'love','I']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as', classifyNB(thisDoc, p0V, p1V, pAb))


'''
====================过滤垃圾邮件====================
'''
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]
def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList) #所有的词汇
    trainingSet= list(range(50)); testSet = [] #python3中要将range转成list
    #随机选取其中的10个作为测试集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainingMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainingMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(trainingMat, trainClasses)
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(wordVector, p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error is:', errorCount/len(testSet))
    return p0V, p1V, pSpam

'''
输入待预测的样本testMail进行预测
'''
def mailTest():
    predictWordList = textParse(open('email/testMail.txt').read())
    docList = [];classList = [];fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # 所有的词汇
    predictMat = bagOfWords2VecMN(vocabList, predictWordList)
    p0V, p1V, pSpam = spamTest()
    result = classifyNB(predictMat, p0V, p1V, pSpam)
    if result == 1:
        print('predict example testMail.txt is a spam email')
    else:
        print('predict example testMail.txt is a ham email')


'''
=======================从个人广告中获取区域倾向=======================
'''
def getDataFromRSS(url):
    data = feedparser.parse(url)
    return data

'''
统计频率最高的30个词
'''
def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {} #里面存放词出现的频率
    for token in vocabList:
        freqDict[token] = fullText.count(token) #count统计fullText里面某个单词出现的次数
    #dict类型的freqDict进行排序后成了list
    sortedFreq = sorted(freqDict.items(), key = operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]



def localWords(feed1, feed0):
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words: #去掉出现频次高的30个词
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(2*minLen)); testSet=[]
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for index in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[index]))
        trainClasses.append(classList[index])
    p0V, p1V, pSpam = trainNB0(trainMat, trainClasses)
    errCount = 0
    for i in testSet:
        if(classifyNB(bagOfWords2VecMN(vocabList,docList[i]), p0V, p1V, pSpam)) != classList[i]:
            errCount += 1
    print('the error rate is: %f' % (errCount/10))
    return vocabList, p0V, p1V

def getTopWords(ny,sf):
    vocabList,p0V, p1V = localWords(ny, sf)
    topNY = []; topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0 :
           topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0 :
           topNY.append((vocabList[i], p1V[i]))
    #print(topSF)
    sortedSF = sorted(topSF, key=lambda pair:pair[1], reverse=True)
    print('SFSFSFSFSFSFSFSFSFSFSFSFSFSFSFSFSFSFSFSFSFSF')
    for i in range(10):
        print(sortedSF[i][0])
    sortedNY = sorted(topNY, key=lambda pair:pair[1], reverse=True)
    print('NYNYNYNYNYNYNYNYNYNYNYNYNYNYNYNYNYNYNYNYNYNY')
    for i in range(10):
        print(sortedNY[i][0])



if __name__ == "__main__":
    # dataSet, classVec = loadDataSet()
    # myVocabList = createVocabList(dataSet)
    # inputSet = dataSet[0]
    # print(setOfWords2Vec(myVocabList,inputSet))
    # trainMatrix = [setOfWords2Vec(myVocabList,i) for i in dataSet]
    # print(trainNB0(trainMatrix, classVec))
    #testNB()
    #spamTest()
    #mailTest()
    #print(getDataFromRSS('https://sfbay.craigslist.org/search/stp?format=rss'))
    #print(getDataFromRSS('https://newyork.craigslist.org/search/stp?format=rss'))
    data1 = getDataFromRSS('https://newyork.craigslist.org/search/stp?format=rss')
    data0 = getDataFromRSS('https://sfbay.craigslist.org/search/stp?format=rss')
    localWords(data1, data0)
    #print(len(vocabList), len(set(fullText)))
    getTopWords(data1,data0)