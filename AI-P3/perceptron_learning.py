# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 17:19:58 2017

@author: philippe.de-cuetos

Perceptron Learning Algorithm
$ python problem1.py input1.csv output1.csv

final values of w_1, w_2, and b will be printed to the output 
struct of output : -13,-6,48
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv)!=3:
    sys.stderr.write("Usage : python %s input_filename output_filename\n" %sys.argv[0])
    raise SystemExit(1)

data=np.genfromtxt(sys.argv[1],delimiter=",")
X=data[:,:-1] 
Y=data[:,-1] 
n=np.shape(X)[0] #nb of feature points
d=np.shape(X)[1] #dimension of feature points

UN=np.ones((1,n)) #python ne semble pas connaître le vecteur colonne 
#add vector 1 to X
Xp=np.insert(X,0,UN,axis=1)

W=np.zeros((1,d+1)) #w0, w1, wd (wo will not be printed) 
B=np.zeros((1,1))

#init figure
plt.figure()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Perceptron')
plt.legend('blue dots : +1 / red dots : -1')

t=0 #current iterative round
while (True):
    t+=1 
    w=np.copy(W[t-1,:]) #current version of weights
    for i in range(n):
        #compute f(xi)
        f_xi=np.sign(np.dot(Xp[i],w)) #compute f(xi)
        if Y[i]*f_xi<=0: #means error
            w=w+Y[i]*Xp[i]  #w and X[i] are d+1 dim vectors
    if np.array_equal(w,W[t-1,:]): #no change in w means convergence
        break
    else:
        W=np.vstack((W,w))
        print("iteration %d weights :" %t,w)                
        #update figure
        
        x_fig = np.linspace(0,15)
        y_fig = -(w[0] + w[1] * x_fig)/w[2]
        plt.plot(x_fig,y_fig)
        for i in range(n):
            if Y[i]>=0: plt.plot(X[i][0],X[i][1],'bo')
            else: plt.plot(X[i][0],X[i][1],'ro')
        plt.show()


#output to dump into file respecting order w1 w2 b
Z=np.hstack((W[:,1:],W[:,0:1]))
np.savetxt(sys.argv[2],Z,fmt='%d',delimiter=',')

