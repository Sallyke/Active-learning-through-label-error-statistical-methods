# python 3.7
import pandas as pd
import numpy as np
import time
from sklearn.metrics import davies_bouldin_score
from numba import jit

global n
global X
global Y
global disMax
global dc
global rho
global master
global delta
global gamma
global DB_index
global e
global c
global totalBuy
global numTeach
global numPredict
global predictedLabels
global BlockInfo

def loadData(path):
    global n
    global X
    global Y
    df = pd.read_excel(path)
    data = np.array(df)
    n, c = data.shape
    X = data[:, 0:c - 1]
    Y = data[:, c - 1].astype(int)
    

def distance(paraI, paraJ ):
    d = np.sqrt(np.sum(np.square(X[paraI]-X[paraJ])))
    return d


def computeMaxdistance():
    global disMax
    disMax = 0
    for i in range(n-1):
        for j in range(i+1, n):
            if disMax < distance(i, j):
                disMax = distance(i, j)

def computeDC(dcrate):
    global dc
    dc = disMax*dcrate

def computeRho():
    global rho
    rho = np.zeros(n, dtype=int)
    for i in range(n-1):
        for j in range(n):
            if distance(i,j) < dc:
                rho[i] = rho[i] + 1
                rho[j] = rho[j] + 1


def computeDelta():
    global master
    global delta

    ordrho = np.argsort(-rho, kind='quickSort')
    delta = np.zeros(n,dtype=float)
    master = np.zeros(n, dtype=int)
    delta[ordrho[0]] = disMax

    for i in range(1,n):
        delta[ordrho[i]] = disMax
        for j in range(i):
            temp=distance(ordrho[i], ordrho[j])
            if temp < delta[ordrho[i]]:
                delta[ordrho[i]] = temp
                master[ordrho[i]] = ordrho[j]


def computeGamma():
    global gamma
    gamma = np.zeros(n)
    for i in range(n):
        gamma[i] = rho[i] * delta[i]


def computeBlock(cc, samples:np.ndarray, flag):
    global DB_index

    if flag == 0:
        priority = np.argsort(-gamma, kind='quickSort')
        centers = priority[0:cc]
        cl = -1 * np.ones(n, dtype=int)
        clusterIndices = np.zeros(n, dtype=int)
        ordrho = np.argsort(-rho, kind='quickSort')
        for i in range(cc):
            cl[centers[i]] = i
        for i in range(n):
            if cl[ordrho[i]] == -1:
                cl[ordrho[i]] = cl[master[ordrho[i]]]
        for i in range(n):
            clusterIndices[i] = centers[cl[i]]

        ## 计算DB指数
        DB_index = davies_bouldin_score(X, cl)

        blockInformation = []
        for i in range(cc):
            tempElements = 0
            for j in range(n):
                if clusterIndices[j] == centers[i]:
                    tempElements = tempElements + 1

            tempblock = np.zeros(tempElements, dtype=int)
            tempElements = 0
            for j in range(n):
                if clusterIndices[j] == centers[i]:
                    tempblock[tempElements] = j
                    tempElements = tempElements + 1

            blockInformation.append(tempblock)

        return centers, blockInformation
    else:

       rho_n = rho[samples]
       ordrho = np.argsort(-rho_n, kind='quickSort')
       ordrho = samples[ordrho]

       gamma_n = gamma[samples]
       priority = np.argsort(-gamma_n, kind='quickSort')
       priority = samples[priority]
       centers = priority[0:cc]

       cl = -1 * np.ones((2,np.shape(samples)[0]), dtype=int)
       cl[0,:] = samples
       clusterIndices = np.zeros(np.shape(samples)[0], dtype=int)
       for i in range(cc):
           for j in range(np.shape(samples)[0]):
               if centers[i]==cl[0,j]:
                   cl[1,j] = i

       for i in range(np.shape(samples)[0]):
           for j in range(np.shape(samples)[0]):
               if ordrho[i] == cl[0,j] and cl[1,j] == -1:
                   temp = master[ordrho[i]]
                   loc = np.where(cl[0,:] == temp)[0]
                   cl[1, j] = findMaster(loc, cl, ordrho[i])

       for i in range(np.shape(samples)[0]):
           clusterIndices[i] = centers[cl[1,i]]

       blockInformation = []
       for i in range(cc):
           tempElements = 0
           for j in range(np.shape(samples)[0]):
               if clusterIndices[j] == centers[i]:
                   tempElements = tempElements + 1

           tempblock = np.zeros(tempElements, dtype=int)
           tempElements = 0
           for j in range(np.shape(samples)[0]):
               if clusterIndices[j] == centers[i]:
                   tempblock[tempElements] = samples[j]
                   tempElements = tempElements + 1

           blockInformation.append(tempblock)
       # centers = samples[centers]
       return centers, blockInformation

#寻找master
def findMaster(loc,cl,paraRho):
    a = cl[1,loc]
    if a != -1:
        return a
    else:
        t = master[cl[0,loc]]
        loc = np.where(cl[0,:] == t)[0]
        return findMaster(loc, cl, paraRho)


def activeLearning(samples):
    global e
    global c
    global percent
    global totalBuy
    global numTeach
    global numPredict
    global predictedLabels
    global BlockInfo

    flag = 0
    tblock = c
    alreadyClassified = -1 * np.ones(n, dtype=int)
    predictedLabels = -1 * np.ones(n, dtype=int)

    # 聚类
    centers, blockInformation = computeBlock(tblock, samples, flag)
    BlockInfo = blockInformation

    while (numTeach < totalBuy):
        need_deal = np.ones(tblock, dtype=bool)
        # 选择关键实例
        critiPionts = np.zeros([tblock, 3], dtype=int)
        for i in range(tblock):
            critiPionts[i][0] = centers[i]
            tempSelect = np.zeros(len(blockInformation[i]))
            for j in range(len(blockInformation[i])):
                tempSelect[j] = distance(centers[i], blockInformation[i][j])
            ordtempSelect = np.argsort(-tempSelect, kind='quickSort')
            critiPionts[i][1] = blockInformation[i][ordtempSelect[0]]

            for j in range(len(blockInformation[i])):
                tempSelect[j] = distance(critiPionts[i][1], blockInformation[i][j])
            ordtempSelect = np.argsort(-tempSelect, kind='quickSort')
            critiPionts[i][2] = blockInformation[i][ordtempSelect[0]]

        # 计算簇直径
        Lambda = np.zeros(tblock)
        for i in range(tblock):
            Lambda[i] = distance(critiPionts[i][1], critiPionts[i][2])/disMax

        # 主动学习
        phi = np.zeros(tblock)
        for i in range(tblock):
            ## 计算误差
            if DB_index <= 1.2:
                phi[i] = 779.9 * (Lambda[i]) ** 9.019 - 0.0006884
            else:
                phi[i] = (804.3 * Lambda[i] - 1.381) / (
                            Lambda[i] ** 3 + 1621 * Lambda[i] ** 2 + 286.2 * Lambda[i] + 1221)

            if phi[i]<=e:
                #购买标签
                for j in range(len(critiPionts[i])):
                    if (alreadyClassified[critiPionts[i][j]]==-1):
                        if numTeach >= totalBuy:
                            break
                        predictedLabels[critiPionts[i][j]] = Y[critiPionts[i][j]]
                        alreadyClassified[critiPionts[i][j]] = 1
                        numTeach+=1
                # 判断纯度
                isUniqueLabel=len(np.unique(predictedLabels[critiPionts[i]]))
                if isUniqueLabel==1:
                    iind = np.where(alreadyClassified[blockInformation[i]]==-1)
                    iind = np.matrix(iind).T
                    predictedLabels[blockInformation[i][iind]] = predictedLabels[critiPionts[i][0]]
                    alreadyClassified[blockInformation[i][iind]] = 1
                    numPredict = numPredict +len(iind)
                    need_deal[i] = False
        # 退出条件
        if np.all(need_deal == False):
            break
        else:
            tblock, centers, blockInformation = splitContinue(need_deal, centers, blockInformation)  # 分裂

        if np.all(need_deal == False):
            break

## 分裂
def splitContinue(need_deal, centers, blockInformation):
    flag = 1
    tempBlock = 0
    centers2 = []
    blockInformation2 = []
    for i in range(len(need_deal)):
        if need_deal[i]:
            if len(blockInformation[i]) < 10:
                need_deal[i] = False
                continue
            centers1, blockInformation1 = computeBlock(2, blockInformation[i], flag)
            centers2.append(centers1)
            for i in range(2):
                blockInformation2.append(blockInformation1[i])
            tempBlock = tempBlock + 2

    if np.all(need_deal == False):
        tblock = len(blockInformation)
        return tblock, centers, blockInformation
    else:
        centers = np.array(centers2)
        centers = centers.flatten()
        centers = centers.T
        blockInformation = blockInformation2
        tblock = tempBlock
        return tblock, centers, blockInformation

def vote():
    temp=np.zeros([c,c],dtype=int)
    for i in range(0,c):
        for j in range(len(BlockInfo[i])):
            for k in range(0,c):
                if predictedLabels[BlockInfo[i][j]]==k+1:
                    temp[i][k]+=1
        Labels =np.argmax(temp[i])
        ind=np.where(predictedLabels[BlockInfo[i]]==-1)
        lo=np.matrix(ind).T
        predictedLabels[BlockInfo[i][lo]] = Labels+1

def getAccuracy():
    errors = 0
    for i in range(n):
        if predictedLabels[i] != Y[i]:
            errors += 1

    accuracy = (n - errors - numTeach) / (n - numTeach)
    return accuracy

def test():
    start = time.time()
    global e
    global c
    global totalBuy
    global numTeach
    global numPredict

    path = r'C:\datasets\excel\irisDecision.xlsx'
    loadData(path)
    print("loadData ok")
    computeMaxdistance()
    print("computeMaxdistance ok")
    dcrate = 0.1  # dcrate
    computeDC(dcrate)
    print("computeDC ok")
    computeRho()
    print("computeRho ok")
    computeDelta()
    print("computeDelta ok")
    computeGamma()
    print("computeGamma ok")
    c = 3  # 类别数
    e = 0.1  # 误差上限
    totalBuy = round(0.1 * n)  # 购买个数
    numTeach = 0
    numPredict = 0
    samples=np.arange(0,n,1)
    activeLearning(samples)
    print("activeLearning ok")
    vote()
    print("vote ok")
    accuracy = getAccuracy()
    print('accuracy:', accuracy)
    end = time.time()
    total_time = end - start
    print("总耗时：" + str(total_time))

if __name__ == '__main__': test()




