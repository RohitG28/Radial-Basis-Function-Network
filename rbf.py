#!/usr/bin/env python
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def gaussianFunction(x, means, sigmaSquared):
	net = [math.exp(-((np.linalg.norm(x-means[i]))**2/sigmaSquared)) for i in range(len(means))]
	net.append(1)
	net = np.array(net)
	return net

def rbf(x,y,noOfHiddenUnits):
	# centerIndices = np.random.choice(x.shape[0], noOfHiddenUnits, replace=False)
	# centers = x[centerIndices,:]
	
	kMeans = KMeans(n_clusters = noOfHiddenUnits)
	kMeans.fit(x)
	centers = kMeans.cluster_centers_
	print(centers)

	maxDist = 0
	for i in range(len(centers)):
		for j in range(i+1,len(centers)):
			tempDist = np.linalg.norm(centers[i]-centers[j])
			if(tempDist > maxDist):
				maxDist = tempDist

	sigmaSquared = (maxDist**2)/noOfHiddenUnits
	nets = [gaussianFunction(x[i],centers,sigmaSquared) for i in range(len(x))]
	nets = np.array(nets)
	weights = np.zeros((noOfHiddenUnits+1,1))
	pseudoInverse = np.matmul(np.transpose(nets),nets)
	pseudoInverse = np.matmul(np.linalg.inv(pseudoInverse),np.transpose(nets))
	weights = np.matmul(pseudoInverse,y)
	return weights, centers, sigmaSquared

def predict(x, weights, centers, sigmaSquared):
	net = gaussianFunction(x,centers,sigmaSquared)
	out = np.matmul(net,weights)
	if out > 0.5:
		plt.plot(x[0],x[1],'x',color = 'blue')
	else:
		plt.plot(x[0],x[1],'ro',color = 'blue')


x11 = np.random.uniform(-4,4,20)
x21 = np.random.uniform(-4,4,20)
x12 = np.random.uniform(-10,10,20)
x22 = np.append(np.random.uniform(-10,-4,10),np.random.uniform(5,10,10)) 

x = []

for i in range(len(x11)):
	x.append(np.array([x11[i],x21[i]]))

for i in range(len(x12)):
	x.append(np.array([x12[i],x22[i]]))

x = np.array(x)
y = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]])

weights, centers, sigmaSquared = rbf(x,y,6)

print(predict(np.array([0,0]),weights,centers,sigmaSquared))
print(predict(np.array([0,1]),weights,centers,sigmaSquared))
print(predict(np.array([-1,2]),weights,centers,sigmaSquared))
print(predict(np.array([10,1]),weights,centers,sigmaSquared))
print(predict(np.array([0,9]),weights,centers,sigmaSquared))
print(predict(np.array([-6,6]),weights,centers,sigmaSquared))
print(predict(np.array([-4,4]),weights,centers,sigmaSquared))
print(predict(np.array([-9,3]),weights,centers,sigmaSquared))
print(predict(np.array([3,3]),weights,centers,sigmaSquared))
print(predict(np.array([2,8]),weights,centers,sigmaSquared))
print(predict(np.array([-4,-6]),weights,centers,sigmaSquared))
print(predict(np.array([-10,6]),weights,centers,sigmaSquared))

# xPlot1 = []
# xPlot2 = []
# yPlot1 = []
# yPlot2 = []

# for i in range(len(x)):
# 	if(y[i][0] == 0):
# 		xPlot1.append(x[i][0])
# 		yPlot1.append(x[i][1])
# 	else:
# 		xPlot2.append(x[i][0])
# 		yPlot2.append(x[i][1])

plt.plot(x11, x21, 'ro', color = 'red')
plt.plot(x12, x22, 'x', color = 'red')

plt.show()

