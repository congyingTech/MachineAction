'''
Created on Nov 22, 2010

@author: Peter
'''
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import CH05SVM.svmMLiA as svm

xcord0 = []
ycord0 = []
xcord1 = []
ycord1 = []
markers =[]
colors =[]
fr = open('testSet.txt')#this file was generated by 2normalGen.py
dataMat, labelMat = svm.loadDataSet('testSet.txt')
dataMatrix = mat(dataMat);labelMatrix =  mat(labelMat).transpose()
for line in fr.readlines():
    lineSplit = line.strip().split('\t')
    xPt = float(lineSplit[0])
    yPt = float(lineSplit[1])
    label = int(lineSplit[2])
    if (label == -1):
        xcord0.append(xPt)
        ycord0.append(yPt)
    else:
        xcord1.append(xPt)
        ycord1.append(yPt)

fr.close()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xcord0,ycord0, marker='s', s=90)
ax.scatter(xcord1,ycord1, marker='o', s=50, c='red')
plt.title('Support Vectors Circled')

#b, alphas, w = svm.smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
b, alphas, w = svm.smoP(dataMatrix, labelMatrix, 0.6, 0.001, 40)



#画出支持向量
sv = [];svCount = 0
for i in range(100):
    if alphas[i]>0.0:
        sv.append(dataMat[i])
        svCount += 1
for i in range(svCount):
    print(sv[i][0], sv[i][1])
for i in range(svCount):
    circle = Circle((sv[i][0], sv[i][1]), 0.5, facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=3,
                    alpha=0.5)
    ax.add_patch(circle)
# circle = Circle((4.6581910000000004, 3.507396), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
# ax.add_patch(circle)
# circle = Circle((3.4570959999999999, -0.082215999999999997), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
# ax.add_patch(circle)
# circle = Circle((6.0805730000000002, 0.41888599999999998), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
# ax.add_patch(circle)
#plt.plot([2.3,8.5], [-6,6]) #seperating hyperplane



b = float(b)
w0 = w[0,0]; w1 = w[0,1]
print(b, w0, w1)
x = arange(-2.0, 12.0, 0.1)
y = (-w0*x - b)/w1
ax.plot(x,y)
ax.axis([-2,12,-8,6])
plt.show()