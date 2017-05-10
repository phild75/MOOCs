# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:35:00 2017

@author: philippe.de-cuetos

ML3 : clustering

start with K-means before doing GMM

5 clusters. Run both algorithms for 10 iteration

initialize the K-means centroids by randomly selecting 5 data points.
outputs :  centroids-[iteration].csv: 
5 rows each containing a centroid 
"""

import sys
import numpy as np
#import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

K=5 #nb of clusters : indexed by 1, ..., K
T=10 #nb of iterations

def Kmeans(X,n,d):
    #initialize the K-means centroids by randomly selecting 5 data points
    mu=np.zeros((K,d))
    rand_points=np.random.randint(n,size=K)
    for i in range(K): 
        m=rand_points[i]
        mu[i,:]=np.copy(X[m])
        #mu[i,:]=np.copy(X[i])    #for testing only : use the 1st 5 rows
    
    
    print("init mu :",mu)
   
    for t in range(T):
        #init cluster association to each point
        C=np.zeros(n)
        #update each ci : cluster that corresponds to closer centroid
        for i in range(n):
            minimum = np.linalg.norm(X[i]-mu[0])**2
            C[i]=1
            for k in range(1,K):
                temp = np.linalg.norm(X[i]-mu[k])**2
                if temp<minimum : 
                    minimum=temp
                    C[i]=k+1  
        
        #update each centroid mu_k
        #values, counts=np.unique(C,return_counts=True) 
        for k in range(K):
            nk=0 #nb of elements in cluster k
            summ=np.zeros(d)
            for i in range(n):
                if C[i]==k+1: 
                    summ+=X[i]
                    nk+=1
            if nk>0: mu[k]=summ/nk
        
        #output to stdout and file
        print("iteration %d: " %t)
        print(mu)
        file_out_name = "centroids-"+str(t+1)+".csv"
        np.savetxt(file_out_name,mu,delimiter=",")
    return(C)


def ME(X,n,d):
    #initializes pi and gaussian parameters
    pi=np.ones(K)/K #vector representing uniform distrib
    
    mean=np.zeros((K,d)) #vector init with 5 random points
    rand_points=np.random.randint(n,size=K)
    for k in range(K): 
        m=rand_points[k]
        mean[k,:]=np.copy(X[m])
    
    cov_dict={} #dict of covariance matrices init to I
    for k in range(K):
        cov_dict.update({k+1:np.identity(d)})
    
    for t in range(T):
        #init cluster association to each point
        C=np.zeros(n)
        
        #E-step : update phi
        phi=np.zeros((n,K)) #row of phi are points index i, colums cluster k
        for i in range(n):
            for k in range(K):
                proba_xi=multivariate_normal.pdf(X[i], mean[k,:], cov_dict[k+1])
                phi[i,k]=pi[k]*proba_xi
            summ = np.sum(phi[i,:])       
            if summ!=0: phi[i,:]=phi[i,:]/summ
        
        #M-step : update pi, mean, cov
        mean=np.zeros((K,d))
        cov_dict={}
        for k in range(K):
            nk=0 #nb of elements in cluster k
            
            for i in range(n):
                mean[k,:]+=phi[i,k]*X[i]
                nk+=phi[i,k]
            mean[k,:]=mean[k,:]/nk
            pi[k]=nk/n
            
            sigma=np.zeros((d,d))
            for i in range(n):
                #sigma+=phi[i,k]*np.dot((X[i]-mean[k,:]).reshape(d,1),(X[i]-mean[k,:]).reshape(1,d))
                sigma+=phi[i,k]*np.outer((X[i]-mean[k,:]),(X[i]-mean[k,:]))
              
            sigma=sigma/nk
            cov_dict.update({k+1:sigma})    
    
        #update cluster association
        for i in range(n):
            C[i]=np.argmax(phi[i,:])+1
    
        #output to stdout and file
        print("---iteration %d: " %t)
        print("pi :",pi)
        print("mean:", mean)
        file_out_name = "pi-"+str(t+1)+".csv"
        np.savetxt(file_out_name,pi,delimiter=",")
        file_out_name = "mu-"+str(t+1)+".csv"
        np.savetxt(file_out_name,mean,delimiter=",")
        for k in range(K):
            file_out_name = "Sigma-"+str(k+1)+"-"+str(t+1)+".csv"
            np.savetxt(file_out_name,cov_dict[k+1],delimiter=",")
                
    return(C)

"""
def draw(data, clusters):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    ax.scatter(x, y, z, c=clusters, marker='.')

    plt.show()
"""
if len(sys.argv)!=2:
    sys.stderr.write("Usage : python %s input_filename \n" %sys.argv[0])
    raise SystemExit(1)

X=np.genfromtxt(sys.argv[1],delimiter=",")
n,d=np.shape(X)

clusters = Kmeans(X,n,d)
print("result of Kmeans:")
#draw(X,clusters)

clusters = ME(X,n,d)
print("result of ME:")
#draw(X,clusters)


    

