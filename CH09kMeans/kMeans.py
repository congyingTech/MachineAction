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

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros([m, 2]))
    centroids = createCent(dataSet, k)
    #print('origin centroids', centroids)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        #为所有点计算簇心，放在clusterAssment中
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:], dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: #簇的分配的簇心如果一直不是最小的index会一直重新分配，直到相等为止
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist**2
        print(centroids)
        #对于归簇的点进行更新簇心
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment


def biMeans(dataSet, k, distMeas = distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros([m, 2]))
    #print(type(mean(dataSet, axis=0)[0,0]))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    #print(centroids)
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0),dataSet[j,:]) ** 2
    while (len(centList)<k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A==i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0],1])
            print('sseSplit, and notSplit', sseSplit, sseNotSplit)
            if(sseNotSplit+sseSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseNotSplit+sseSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        centList[bestCentToSplit] = bestNewCents[0, :]
        centList.append(bestNewCents[1,:])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss
    return mat(centList), clusterAssment


if __name__ == '__main__':
    datMat = mat(loadDataSet('testSet.txt'))
    #centroids, clusterAssment = kMeans(datMat, 4)
    #print('result========',centroids, clusterAssment)
    biMeans(datMat,2)