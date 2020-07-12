import numpy as np
import pandas as pd
from numpy import *
import random
import matplotlib.pyplot as plt


#数据读取
def getData(filename):
	df = pd.read_excel(filename)
	data = np.array(df.values[:,1:])
	return data

#计算样本距离，这里采用欧式距离
def calDistance(dataI, dataJ):
	subdu = dataI - dataJ
	# subdu.shape = (1,subdu.shape[0])
	dist = np.dot(subdu, subdu.T)**0.5
	return dist

#k均值聚类实现
def kMeansCluster(data, k, numIt):
	m, n = np.shape(data)
	Mu = np.zeros((k, n))
	randomList = random.sample(range(m), k)
	for i in range(k):
		Mu[i] = data[randomList[i]]
	for num in range(numIt):
		Ci = {}
		oldMu = Mu.copy()
		label = np.zeros((m,1))
		for j in range(m):
			minDist = inf
			distIJ = np.zeros((k))
			for i in range(k):
				distIJ[i] = calDistance(Mu[i], data[j])
			# print(distIJ)
			ListDist = distIJ.tolist()
			minDist = min(ListDist)
			label[j] = ListDist.index(minDist)
		labelList = label.flatten().tolist()
		for i in range(k):
			MuI = [0.0, 0.0]
			for j in range(m):
				if label[j] == i: MuI += data[j]
			MuI = MuI/labelList.count(i)
			if Mu[i].all() != MuI.all(): 
				Mu[i] = MuI
			Mu[i] = MuI
		if Mu.tolist() == oldMu.tolist(): 
			print('迭代次数:',num)
			break
	return Mu, label

		

trainSet = getData('watermelon4.0.xlsx')
labelCenter, dataLabel = kMeansCluster(trainSet, 3, 5)

#画图
f1 = plt.figure(1)
plt.subplot(111)
plt.scatter(labelCenter[:,0], labelCenter[:,1], marker = '+', color = 'b', s = 100)
plt.scatter(trainSet[:,0], trainSet[:,1])
m, n = np.shape(trainSet)
M, N = np.shape(labelCenter)
for i in range(M):
	for j in range(m):
		if dataLabel[j] == i: plt.plot([labelCenter[i, 0], trainSet[j, 0]], [labelCenter[i, 1], trainSet[j, 1]], color = 'r')
plt.show()
