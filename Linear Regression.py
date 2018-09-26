import numpy as np
a=[i for i in range(1,6)]
b=[i for i in range(2,7) ]

x1=np.array(a).reshape(1,len(a))
y=np.array(b).reshape(1,len(b))

theta=np.random.randint(-2,high=2,size=2).reshape(1,2)

def train(x,y,alpha,m):
	global theta
	prediction=np.matmul(theta,x)
	error=y-prediction

	gradient=(alpha/m)*np.matmul((prediction-y),x.T)

	theta=theta-gradient

def loss(x,y,m):
	prediction=np.matmul(theta,x)
	error=y-prediction

	return np.sum(np.power(np.subtract(prediction,y),2)/(1/2*m))
m=y.size
X1=np.vstack((np.ones(m).reshape(1,m),x1))
for i in range(1000):
	train(X1,y,0.001,m)
	if(i%100==0):
		loss1=loss(X1,y,m)
		print("Loss=",loss1)

print(theta)
