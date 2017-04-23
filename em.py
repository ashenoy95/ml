
#%matplotlib inline
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math

MOG = 3
prob_c = np.ones(MOG)*(1/MOG) 
mu = np.random.random(MOG)*50 
sigma = np.random.random(MOG)*100
X = np.loadtxt('data1.txt')
#data2 = np.loadtxt('data2.txt')
#data3 = np.loadtxt('data3.txt')
Nj = len(X)
prob_ij = np.zeros((MOG, Nj)) 
ij_hat = np.zeros((MOG, Nj)) 
likelihood = []
delta = 0.00001
ctr = 0

while True: 
	ctr += 1
	
	#E Step
	for i in range(MOG):
		prob_ij[i] = prob_c[i] * mlab.normpdf(X, mu[i], sigma[i])
	prob_x = np.sum(prob_ij, axis=0) #P(X_j)
	prob_ij = prob_ij / prob_x
	Ni = np.sum(prob_ij, axis=1)

	#M Step
	mu = np.sum(prob_ij*X, axis=1)/Ni
	mu_x_square = np.sum(prob_ij*(np.square(X)), axis=1)/Ni
	sigma = np.sqrt(np.sum(prob_ij * pow(X - mu[:,np.newaxis],2), axis=1) / np.sum(prob_ij, axis=1)) #E[(x-mu)^2] 
	prob_c = Ni/Nj #new class prior: proportion of weighted samples attributed to class

	#Log likelihood
	for i in range(MOG):
		ij_hat[i] = prob_c[i] * mlab.normpdf(X, mu[i], sigma[i])
	x_hat = np.sum(ij_hat, axis=0)
	L = 0
	for j in range(Nj):
		L += math.log(x_hat[j])
	print("iter:",ctr,"L:",L)
	likelihood.append(L)
	if ctr==200:
		break
	#if (L-likelihood[-1])<delta:
	#	break
    
print ("Mean:",mu,"\nSD:",sigma,"\nWeights:",prob_c)
plt.plot(likelihood)
plt.show()




