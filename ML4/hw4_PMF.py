#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 07:51:58 2017

@author: phildec

ML P.4 : probabilistic matrix factorization (PMF) for collaborative filtering
-> use MAP inference coordinate ascent algo 

python hw4_PMF.py ratings.csv

ratings.csv: A comma separated file containing the data. 
Each row contains a three values that correspond in order to: user_index, object_index, rating

"""

import sys
import numpy as np

#fixed parameters
T=50 #nb of iterations
d=5 #dimension of decomposed vectors
sig2=0.1 
lamb=2 #lambda

Tprint=[10,25,50] #iterations when stats are printed out

"""  
if len(sys.argv)!=2:
    sys.stderr.write("Usage : python %s input_filename \n" %sys.argv[0])
raise SystemExit(1)
"""
    
input_filename=sys.argv[1]
    
X=np.genfromtxt(input_filename,delimiter=",")

n_entries=np.shape(X)[0] #nb of entries in ratings.csv
n_u=int(np.max(X[:,0]))  #n_u : nb of users indexed : 1,...,n_u
n_v=int(np.max(X[:,1])) #n_v : nb of objects indexed : 1,...,n_v 
r  =int(np.max(X[:,2])) #max rating : 1,...,r
    
#construct matrix M
M=np.zeros((n_u,n_v))
for i in range(n_entries):
    M[int(X[i,0]-1),int(X[i,1]-1)]=X[i,2] #matrix indexes starts at 0    

#decomposition matrices
U=np.zeros((n_u,d))
V=np.zeros((d,n_v))

#PMF objective fct :
L=np.zeros(T)

U_print=list()
V_print=list()

#init V : gaussian
mu=np.zeros(d)
covar=np.identity(d)/lamb 
for j in range(n_v):
    V[:,j]=np.random.multivariate_normal(mu,covar,1)

#Om_ui: objects rated by user i
Om_ui=list()
for i in range(n_u):
    Om_ui.append(M[i,:].nonzero()[0])

#Om_vj: user having rated object j
Om_vj=list()
for j in range(n_v):
    Om_vj.append(M[:,j].nonzero()[0])

for t in range(1,T+1):
    
    #update user location
    for i in range(n_u):
        sum1=np.zeros((d,d))
        sum2=np.zeros(d)
        for j in list(Om_ui[i]):
            sum1+=np.outer(V[:,j],V[:,j])
            sum2+=M[i,j]*V[:,j]
        inv=np.linalg.inv(lamb*sig2*np.identity(d)+sum1)
        U[i,:]=np.dot(inv,sum2)
        
    
    #update object location
    for j in range(n_v):
        sum1=np.zeros((d,d))
        sum2=np.zeros(d)
        for i in list(Om_vj[j]):
            sum1+=np.outer(U[i,:],U[i,:])
            sum2+=M[i,j]*U[i,:]
        inv=np.linalg.inv(lamb*sig2*np.identity(d)+sum1)
        V[:,j]=np.dot(inv,sum2)
    
    #compute objective fct L 
    s1=0
    for i in range(n_u):
        for j in list(Om_ui[i]):
            s1+=(M[i,j]-np.dot(U[i,:],V[:,j]))**2
    s1=s1/(2*sig2)
    
    s2=0
    for i in range(n_u):
        s2+=np.dot(U[i,:],U[i,:])
    s2=s2*lamb/2
    
    s3=0
    for j in range(n_v):
        s3+=np.dot(V[:,j],V[:,j])
    s3=s3*lamb/2

    L[t-1]=-s1-s2-s3
    #print("t = %d : L = %f" %(t,L[t-1]))
    
    #save stats to print out 
    if t in Tprint:
        #keep u and v
        U_print.append(U.copy())
        V_print.append(V.T.copy())
       
#print stats
for t in Tprint:
    np.savetxt("U-"+str(t)+".csv",U_print.pop(0),delimiter=",")
    np.savetxt("V-"+str(t)+".csv",V_print.pop(0),delimiter=",")

np.savetxt("objective.csv",L,delimiter=",")


    
    
    
