from numpy import *
import operator

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines() # 读取所有行
    numberOfLines = len(arrayOLines) # 得到文件行数
    returnMat = zeros((numberOfLines,3)) # 创建返回的numpy矩阵
    classLabelVector = [] # 创建返回的分类标签向量
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t') # 将每行数据分割成一个列表
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]) # 4*2
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 创建diffMat矩阵
    diffMat = tile(inX,(dataSetSize,1)) - dataSet 
    # 计算各个点的欧氏距离
    sqDiffMat = diffMat**2 # 4*2
    sqDistances = sqDiffMat.sum(axis=1) # 4*1
    distances = sqDistances**0.5 # 4*1
    # 排序
    sortedDistIndicies = distances.argsort() # 4*1
    classCount = {}
    # 寻找最近的3个点，看哪个lable最多
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] # A
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 # A:1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) # [('A',2)]
    return sortedClassCount[0][0]

group,labels = createDataSet()
print(classify0([0,0],group,labels,3))
