#Aniket Shenoy
#ashenoy@iu.edu
#python3

import numpy as np
import matplotlib.pyplot as plt

def phi(v):										#sigmoid func
	return 1/(1+np.exp(-v))

def backprop(eta):									#online backprop
	#i/p truth table for 4 bit parity checker
	X = np.array([	[0,0,0,0,1],
					[0,0,0,1,1],
					[0,0,1,0,1],
					[0,0,1,1,1],
					[0,1,0,0,1],
					[0,1,0,1,1],
					[0,1,1,0,1],
					[0,1,1,1,1],
					[1,0,0,0,1],
					[1,0,0,1,1],
					[1,0,1,0,1],
					[1,0,1,1,1],
					[1,1,0,0,1],
					[1,1,0,1,1],
					[1,1,1,0,1],
					[1,1,1,1,1]])
	D = np.array([[0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0]]).T 				#desired o/p

	#randomly initialized weights b/w -1 & 1 (0 mean)
	w_ij = np.random.uniform(-1,1,(4,5))
	w_jk = np.random.uniform(-1,1,(1,5))

	delta_w_ij = np.zeros((4,5))
	delta_w_jk = np.zeros((1,5))

	epoch = 0
	alpha = 0.9											
	print('\neta:',eta)
	
	while True:
		diff = []								#stores abs error (diff) for all i/p patterns in current epoch	
		for i in range(X.shape[0]):						#online update
			x = np.reshape(X[i,],(5,1)) 					#single i/p pattern
			d = D[i]							#desired o/p corresponding to i/p pattern

			#i: i/p layer
			#j: hidden layer
			#k: o/p layer

			#feed-forward
			v_j = w_ij.dot(x)
			y_j = phi(v_j)
			y_j = np.append(y_j,[[1]],axis=0)				#adding +1 bias term
			v_k = w_jk.dot(y_j)
			y_k = phi(v_k)
			e_k = d - y_k

			#backprop 
			delta_k = e_k*y_k*(1-y_k)					#delta rule
			delta_j = w_jk.T*delta_k*y_j*(1-y_j)
			
			#weight updation
			delta_w_jk = eta*delta_k.dot(y_j.T) + alpha*delta_w_jk
			delta_w_ij = eta*delta_j[:-1].dot(x.T) + alpha*delta_w_ij
			w_ij+= delta_w_ij
			w_jk+= delta_w_jk

			diff.append(e_k[0,0])						#appending error for current i/p pattern

		epoch+=1								#epoch complete
		error = np.max(np.abs(diff))						#abs error (diff) in current epoch
		if epoch%10000==0:							#printing error after every 10000 epochs
			print('epoch:',epoch,'\nerror:',error,'\n')
		if error<=.05:								#convergenve criteria
			break
		#np.random.shuffle(X)							#shuffle i/p patterns for next epoch (helps to avoid local minima)

	print('epoch:',epoch,'\nerror:',error,'\n')
	return epoch


etas = np.arange(.05,.55,.05)								#range of learning rates from .05 to .5 with a .05 increment
epochs = list(map(backprop,etas))							#call the backprop function for each of the etas in the range
print('\n')
for eta, n in zip(etas,epochs):
	print('eta:',eta)
	print('epoch:',n,'\n')

#plot etas vs. epochs for the eta range
plt.plot(etas,epochs)
plt.xlabel('eta')
plt.ylabel('epochs')
plt.show()

