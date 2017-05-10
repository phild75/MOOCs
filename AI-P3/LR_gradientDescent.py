# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 17:14:51 2017

@author: philippe.de-cuetos

Linear Regression with gradient descent

input shape : 2,10.21027,0.8052476 2.04,13.07613,0.9194741  ...
(age,weight, height) age(years),weight(kg), height(meters)=label

Scale each feature (i.e. age and weight) by its standard deviation, 
and set its mean to zero. 

Gradient Descent. Implement gradient descent to find a regression model                   
learning rates: α ∈ {0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10}
For each value of α, run the algorithm for exactly 100 iterations

output : $ python3 problem2_3.py input2.csv output2.csv

ten cases in total, nine with the specified learning rates (and 100 iterations), 
and one with your own choice of learning rate (and your choice of number of iterations)

outputfile : alpha, number_of_iterations, b_0, b_age, and b_weight 
expressed with as many decimal places as you please.


"""
import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def gradientDescent(alpha,iter,X,Y,d,n):
    beta=np.zeros((1,d+1))
    
    for t in range(iter): #for each iteration
        for j in range(d+1): #for each beta dimension b
            b=beta[0][j]
            S=0
            for i in range(n):
               S+=(X[i,:].dot(beta.transpose())-Y[i])*X[i,j] 
            b=b-alpha*S/n
            beta[0][j]=b    
    
    #compute final risk :
    risk=np.sum((X.dot(beta.transpose())-Y)**2)/n/2
    return(risk,beta)


#main program 
if len(sys.argv)!=3:
    sys.stderr.write("Usage : python %s input_filename output_filename\n" %sys.argv[0])
    raise SystemExit(1)

data=np.genfromtxt(sys.argv[1],delimiter=",")

X=data[:,:-1] #age and weight
Y=data[:,-1] #Y is the height
n=np.shape(X)[0] #nb of feature points
d=np.shape(X)[1] #dimension of feature points

UN=np.ones((1,n))
Xp=np.copy(X) 
Xp=np.insert(Xp,0,UN,axis=1)

#normalize x1 and x2
for i in range(d):
    mean = np.mean(Xp[:,i+1])
    std = np.std(Xp[:,i+1])
    Xp[:,i+1]=(Xp[:,i+1]-mean)/std

Alpha=np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.045]) 
Alpha=Alpha.reshape(-1,1)#make Alpha a column vector :

Iter=np.ones((10,1))*100
Iter[9,0]=110
Iter=Iter.astype(int)
Beta=np.zeros((10,d+1))
Risk=np.zeros((10,1)) #final risk
             
#for each alpha, implement gradient descent to determine Beta
for i in range(10):
    Risk[i],Beta[i]=gradientDescent(Alpha[i][0],Iter[i][0],Xp,Y,d,n)

Z=np.hstack((Alpha,Iter,Beta))
print(Risk)
print(Z)
#TODO: format output : 0.001,100,[b_0],[b_age],[b_weight]
#for i in range(10):
#np.savetxt(sys.argv[2],Z,fmt='%3.4f',delimiter=',')

file_out = open(sys.argv[2],"w")
for i in range(10):
    print("%.4f,%d,%e,%e,%e" %(Z[i,0],Z[i,1],Z[i,2],Z[i,3],Z[i,4]), file=file_out)    
file_out.close()

#Plot graph :
#init figure
"""
#could not do it => do it later 
for i in range(10):
    print("%.4f,%d,%e,%e,%e" %(Z[i,0],Z[i,1],Z[i,2],Z[i,3],Z[i,4]))
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('age')
    ax.set_ylabel('weight')
    ax.set_zlabel('height')
    ax.scatter(Xp[:,1],Xp[:,2],Y)
    
    #todo : create mesh with rights span
    xx, yy = np.meshgrid(range(10), range(10,5,50))
    #compute z for each point
    Xprime=np.hstack((UN,xx,yy))
    zz=X.dot(Beta[i].transpose())
    ax.plot_surface(xx,yy,zz)
    plt.show()
    #for j in range(n):
    #    plt.plot(X[j,0],X[j,1],Y[j])

"""
