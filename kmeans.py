import numpy as np
import pandas as pd

# 计算两个向量的欧式距离，结果为标量
def distEcludScalar(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2))) # la.norm(vecA-vecB)

# k-means 聚类算法
# 该算法会创建k个质心，然后将每个点分配到最近的质心，再重新计算质心。这个过程重复数次，直到数据点的簇分配结果不再改变为止。
def kMeans(dataSet, k, distMeas=distEcludScalar):
    # 为给定数据集构建一个包含k个随机质心的集合。
    def createRandCent(dataSet, k):
    #    np.random.seed(1)
        n = dataSet.shape[1] # 列的数量
        centroids = np.mat( np.zeros((k,n)) ) # 创建k个质心矩阵
        for j in range(n): # 创建随机簇质心，并且在每一维的边界内
            minJ = np.min(dataSet[:,j]) 
            rangeJ = float(np.max(dataSet[:,j]) - minJ) 
            centroids[:,j] = minJ + rangeJ * np.random.rand(k,1)
        return centroids
    
    m = dataSet.shape[0]    # 行数
    clusterAssment = np.mat(np.zeros((m, 2)))    # 创建一个与 dataSet 行数一样，但是有两列的矩阵，用来保存簇分配结果（一列簇索引值、一列误差）
    centroids = createRandCent(dataSet, k)    # 创建质心，随机k个质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):    # 循环每一个数据点并分配到最近的质心中去
            minDist = np.inf; minIndex = -1
            for j in range(k): #1.寻找最近的质心
                distJI = distMeas(centroids[j,:],dataSet[i,:])    # 计算数据点到质心的距离
                if distJI < minDist:    # 如果距离比 minDist（最小距离）还小，更新 minDist（最小距离）和最小质心的 index（索引）
                    minDist = distJI; minIndex = j
            if clusterAssment[i, 0] != minIndex:    # 簇分配结果改变
                clusterChanged = True    # 簇改变
                clusterAssment[i, :] = minIndex, minDist**2    # 更新簇分配结果为最小质心的 index（索引），minDist（最小距离）的平方
 #       print(centroids)
        for cent in range(k): #2.更新质心的位置
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A==cent)[0]] # 获取该簇中的所有点
            centroids[cent,:] = np.mean(ptsInClust, axis=0) # 将质心修改为簇中所有点的平均值，mean 就是求平均值的
        #处理nan
        centroids = np.nan_to_num(centroids)
    return centroids, clusterAssment

# 计算两个向量的欧式距离，结果为列向量
def distEcludColumnVector(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2) ,axis=1))

# 获得kmeans的特征列
def getKmeansDf(dataSet, centroids, k, distMeas=distEcludColumnVector):
    m = dataSet.shape[0]    # 行数
    df = pd.DataFrame(np.arange(m), columns=["temp"])
    for j in range(k): #k为质心数
        a = distMeas(centroids[j,:], dataSet)    # 计算数据点到各个质心的距离
        df["kmeans"+str(j+1)] = pd.DataFrame(a)
    df = df.drop(["temp"], axis=1)
    return df
    
def createKmeansFeature(dataSet, usecols, k):
    dataSet = dataSet[:][usecols] #选取特定的列
    centroids, clusterAssment = kMeans(np.array(dataSet),k) #kMeans聚类
    df = getKmeansDf(np.array(dataSet), centroids, k) #获取每个样本到k个中心样本的距离
    return df

''' createKmeansFeature()函数使用示例 '''
if __name__ == "__main__":
    # 使用哪些特征：使用的列必须都是数字
    usecols = [
        'quoting_attack_level',
        'third_party_attack_level',
        'other_attack_level',
        'attack_level',
        'topic0',
        'topic1',
        'topic2',
        'topic3',
        'topic4',
        'topic5',
        'topic6',
        'topic7'
    ]
    
    from input import read_dataset
    train = read_dataset('clean_train.csv')
    train = train[:100][:]
    df = createKmeansFeature(train, usecols, 6) #调用编写的函数
    dataSet = pd.concat([train, df], axis=1) #拼接原始特征与新生成的特征
    print(dataSet)