# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:41:07 2017

@author: philippe.de-cuetos

 learn a classification model for a chessboard-like dataset

- Input data form :
A,B,label
3.4,0.35,1
0.7,0.2,1
3.55,2.5,0

use scikit-learn

- Make a scatter plot of the dataset showing the two classes with two different patterns.

- Use SVM with different kernels to build a classifier.
- use stratified sampling (i.e. same ratio of positive to negative in both the training and testing datasets)
- Use cross validation (with the number of folds k = 5) instead of a validation set. 

- SVM with Linear Kernel : try values of C = [0.1, 0.5, 1, 5, 10, 50, 100]
- SVM with Polynomial Kernel. Try values of C = [0.1, 1, 3], degree = [4, 5, 6], and gamma = [0.1, 1].
- SVM with RBF Kernel. Try values of C = [0.1, 0.5, 1, 5, 10, 50, 100] and gamma = [0.1, 0.5, 1, 3, 6, 10].
- Logistic Regression. Try values of C = [0.1, 0.5, 1, 5, 10, 50, 100].
- k-Nearest Neighbors. Try values of n_neighbors = [1, 2, 3, ..., 50] and leaf_size = [5, 10, 15, ..., 60].
- Decision Trees. Try values of max_depth = [1, 2, 3, ..., 50] and min_samples_split = [1, 2, 3, ..., 10].
- Random Forest. Try values of max_depth = [1, 2, 3, ..., 50] and min_samples_split = [1, 2, 3, ..., 10].

- training data -> best score
- testing data -> test score

- Ouput file form :
svm_linear,[best_score],[test_score]
svm_polynomial,[best_score],[test_score]
svm_rbf,[best_score],[test_score]
logistic,[best_score],[test_score]
knn,[best_score],[test_score]
decision_tree,[best_score],[test_score]
random_forest,[best_score],[test_score]

"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def printresults(file_out, method, mean_score, best_score, test_score):
    print("-----",method)
    print("mean test score: ", mean_score)
    print("best score: ", best_score)
    print("test_score: ", test_score)
    print("%s,%.4f,%.4f" %(method, best_score, test_score), file=file_out)
    return

#main program
if len(sys.argv)!=3:
    sys.stderr.write("Usage : python %s input_filename output_filename\n" %sys.argv[0])
    raise SystemExit(1)

data=np.genfromtxt(sys.argv[1],delimiter=",") 
#je n'arrive pas à skipper le header sur le fichier initial bizarrement
#donc modifié, tant pis 

file_out = open(sys.argv[2],"w")

Xin=data[:,0:-1]
y=data[:,-1].astype(int) #binary classification 
#beware that Y is 1d vector -> line 
n=np.shape(Xin)[0]
d=np.shape(Xin)[1]

#scale X :
X = preprocessing.scale(Xin) #improve performance significantly !
#plot data
"""
plt.figure()
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend('blue dots : +1 / red dots : 0')
for i in range(n):
    if Y[i]>0: plt.plot(X[i][0],X[i][1],'bo')
    else: plt.plot(X[i][0],X[i][1],'ro')
plt.show()
"""

#stratified sampling with 40% testing data
X_train, X_test, y_train, y_test = train_test_split(\
   X,y,test_size=0.4,stratify=y,random_state=0) 

#list of dictionaries 

SVM_parameters = [ {'kernel': ['linear'], 'C': [0.1, 0.5, 1, 5, 10, 50, 100]},
                    {'kernel': ['poly'], 'C': [0.1, 1, 3], 'degree': [4, 5, 6], 'gamma': [0.1, 1]},
                     {'kernel': ['rbf'], 'gamma': [0.1, 0.5, 1, 3, 6, 10], 'C': [0.1, 0.5, 1, 5, 10, 50, 100]}]

#SVM_parameters = [{'kernel': ['poly'], 'C': [3], 'degree': [6]}]

for i in range(len(SVM_parameters)):
    svr = svm.SVC()
    clf = GridSearchCV(svr, SVM_parameters[i], cv=5) 
    """for each model learning given a param : 
    does k-fold cross-validation with k=5 and compute mean score : training best score
    """ 
    clf.fit(X_train, y_train)
    test_score=clf.score(X_test,y_test)
    if i==0:
        printresults(file_out,"svm_linear",clf.cv_results_["mean_test_score"],clf.best_score_,test_score)
    elif i==1:
        printresults(file_out,"svm_polynomial",clf.cv_results_["mean_test_score"],clf.best_score_,test_score)
    else:
        printresults(file_out,"svm_rbf",clf.cv_results_["mean_test_score"],clf.best_score_,test_score)

#logistic regression : 
LR_parameters = [{'C': [0.1, 0.5, 1, 5, 10, 50, 100]}]
lr = LogisticRegression()
clf = GridSearchCV(lr, LR_parameters, cv=5) 
clf.fit(X_train, y_train)
test_score=clf.score(X_test,y_test)
printresults(file_out,"logistic",clf.cv_results_["mean_test_score"],clf.best_score_,test_score)

#K-NN
kNN_parameters = [{'n_neighbors': list(range(1,51)), 'leaf_size' : [x*5 for x in range(1,13)]}]
knn = KNeighborsClassifier()
clf = GridSearchCV(knn, kNN_parameters, cv=5) 
clf.fit(X_train, y_train)
test_score=clf.score(X_test,y_test)
printresults(file_out,"knn",clf.cv_results_["mean_test_score"],clf.best_score_,test_score)

#same set of params for Decision Trees or Random Forests :
Tree_parameters = [{'max_depth': list(range(1,51)), 'min_samples_split' : list(range(2,11))}]
#keep value 1 for min_samples_split ?

clf = DecisionTreeClassifier()
clf = GridSearchCV(clf, Tree_parameters, cv=5) 
clf.fit(X_train, y_train)
test_score=clf.score(X_test,y_test)
printresults(file_out,"decision_tree",clf.cv_results_["mean_test_score"],clf.best_score_,test_score)

clf = RandomForestClassifier()
clf = GridSearchCV(clf, Tree_parameters, cv=5) 
clf.fit(X_train, y_train)
test_score=clf.score(X_test,y_test)
printresults(file_out,"random_forest",clf.cv_results_["mean_test_score"],clf.best_score_,test_score)


file_out.close()
"""


#linear SVC
clf = svm.SVC()
clf.set_params(C=0.1,kernel='linear').fit(X_train,y_train)
scores = cross_val_score(clf, X_train, y_train, cv=5)
train_score=scores.mean()
test_score=clf.score(X_test,y_test)

print("train score = %.3f , test score = %.3f" %(train_score,test_score) )
"""
#file_out = open(sys.argv[2],"w")

