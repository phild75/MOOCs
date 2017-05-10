#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:01:11 2017

@author: phildec

ML W3 project : hw1_regression.py

Implementing ridge and active learning linear regression
Part 1. -> outputs wRR for arbitrary lambda (wRR_[lambda].csv)
Part 2. -> implements active learning for 1st 10 new data taken from
    the test set and outputs index of these data


python hw1_regression.py lambda sigma2 X_train.csv y_train.csv X_test.csv

"""

import sys
import numpy as np
T=10 #nb of iterations 

def ridgeRegression(X, y, lam):
    #considers X with added 1 in the *last* column
    #1st assumes no centering neither standardizing
    n,d=np.shape(X)
    I=lam*np.identity(d)
    temp=I+np.dot(X.T,X)
    temp2=np.linalg.inv(temp)
    temp3=np.dot(temp2,X.T)
    y=y.reshape(n,1) #write y as column vector
    wRR=np.dot(temp3,y)
       
    return(wRR)

def activeLearning(X, y, X0, lam, s2, wRR):
    index=[0 for i in range(T)]#stores indexes of chosen values for x0 from X0
    n,d=np.shape(X)
    n_test=np.shape(X0)[0]
    y=y.reshape(n,1) #write y as column vector
    
    #compute mu and S
    S=np.linalg.inv(lam*np.identity(d)+np.dot(X.T,X)/s2)
    temp=np.linalg.inv(lam*s2*np.identity(d)+np.dot(X.T,X))
    mu=np.dot(np.dot(temp,X.T),y) #column vector
    
    for t in range(T):
        #look for unmeasured data with highest covariance
        stats=np.zeros((n_test,2)) #stores x0, s02 for each value of X0
        for i in range(n_test):
            x0=X0[i,:] #line vector
            mu0=float(np.dot(x0,mu))
            s02=s2+float(np.dot(np.dot(x0,S),x0.reshape(d,1)))
            stats[i,0]=mu0
            stats[i,1]=s02
        index[t]=np.argmax(stats[:,1])
        #extract x0 from X0 and measure y0
        x0=np.copy(X0[index[t]])
        X0[index[t]]=0
        mu0,s02=stats[index[t]]
        y0=float(np.dot(x0,wRR)) #or =mu0 ?
        
        #update X and y, and the posterior parameters
        X=np.vstack((X,x0))
        y=np.vstack((y,y0))
        n+=1
        
        S=np.linalg.inv(lam*np.identity(d)+np.dot(X.T,X)/s2) #can be simplified
        temp=np.linalg.inv(lam*s2*np.identity(d)+np.dot(X.T,X))
        mu=np.dot(np.dot(temp,X.T),y)
        
    return(index)


#main
if len(sys.argv)!=6:
    sys.stderr.write("Usage : python %s lambda sigma2 X_train.csv y_train.csv X_test.csv\n" %sys.argv[0])
    raise SystemExit(1)
    
lamb=int(sys.argv[1])
sigma2=int(sys.argv[2])
X_train_filename=sys.argv[3]
y_train_filename=sys.argv[4]
X_test_filename=sys.argv[5]

X_train=np.genfromtxt(X_train_filename,delimiter=",")
y_train=np.genfromtxt(y_train_filename,delimiter=",")
X_test=np.genfromtxt(X_test_filename,delimiter=",")


wRR=ridgeRegression(X_train, y_train, lamb)
index=activeLearning(X_train, y_train, X_test, lamb, sigma2, wRR)


#output 
file_out_name = "wRR_"+str(lamb)+".csv"
np.savetxt(file_out_name,wRR,fmt="%f",delimiter="\n")
 
index_out=str(index[0]+1)
for i in range(1,len(index)):
    index_out+=","+str(index[i]+1)
    
file_out_name = "active_"+str(lamb)+"_"+str(sigma2)+".csv"
file_out=open(file_out_name,"w")
print(index_out,file=file_out) #python 2 : print >>file_out,index_out
file_out.close()

    
    