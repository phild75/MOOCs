# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:42:18 2017

@author: philippe.de-cuetos

Beware: only testing file, not submission file  
ML W2 project : hw2_classification.py

Assume labeled data dimensiond, K-class Bayes classifier
-> ML estimates for class prior proba and class-specific Gaussian parms for each class
-> run new predictions

python hw2_classification.py X_train.csv y_train.csv X_test.csv

Outputs : probs_test.csv: csv file containing 
the posterior probabilities P(Y=y|X=x) of the label of each row in "X_test.csv".

"""
import numpy as np
import sys
from scipy.stats import multivariate_normal

global K #nb of classes : indexed 0,...,9


if __name__ == "__main__":
    
    if len(sys.argv)!=4:
        sys.stderr.write("Usage : python %s X_train.csv y_train.csv X_test.csv\n" %sys.argv[0])
        raise SystemExit(1)
    
    K=10 
    
    #get data 
    X_train=np.genfromtxt(sys.argv[1],delimiter=",")
    y_train=np.genfromtxt(sys.argv[2],delimiter=",")
    X_test=np.genfromtxt(sys.argv[3],delimiter=",")
    
    n,d=np.shape(X_train)
    n_test=np.shape(X_test)[0]
    #K=int(np.max(y_train))+1 #supposes K index starts at 0
    
    #class priors 
    pi=list()
    for k in range(K):
        mle=np.shape(y_train[y_train==k])[0]/float(n) #mean nb of elements ==i
        pi.append(mle)
    
    print(pi)
    
    #class conditional density Gaussian parameters
    mu=np.zeros((K,d))
    sigma=[np.zeros((d,d)) for i in range(K)]
    for k in range(K):
        ny=pi[k]*n #nb elements of class i
        for i in range(n):
            if y_train[i]==k: mu[k,:]+=X_train[i,:]
        if ny>0: mu[k,:]=mu[k,:]/ny
        for i in range(n):
            if y_train[i]==k: sigma[k]+=np.outer(X_train[i,:]-mu[k,:],X_train[i,:]-mu[k,:])
        if ny>0: sigma[k]=sigma[k]/ny
    
    print(mu)
    print(sigma)
    
    posterior=np.zeros((n_test,K))
    for i in range(n_test):
        for k in range(K):
            if sigma[k].all()!=0: #in case sigma is null matrix
                proba_y=multivariate_normal.pdf(X_test[i], mu[k,:], sigma[k])
            posterior[i,k]=proba_y*pi[k]
        if np.sum(posterior[i,:])>0:
            posterior[i,:]/=np.sum(posterior[i,:]) #normalize to get probabilities
    
    print(posterior)    
    
    #output to file:
    np.savetxt("probs_test.csv",posterior,delimiter=",")
        
        
        
        
        
        