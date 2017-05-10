train_path = "../resource/asnlib/public/aclImdb/train/" # use terminal to ls files under this directory
test_path = "../resource/asnlib/public/imdb_te.csv" # test data for grade evaluation

import numpy as np
import pandas as pd 
import sys
import os
import codecs
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV


global stopwset
stopwset=set()

def remove_common_stopwords(line,stopwords):
    
    #remove punctuation for convenience
    line=line.replace(",","")
    line=line.replace("?","")
    line=line.replace("!","")
    line=line.replace(":","")
    line=line.replace(".","")
    line=line.replace(";","")
    line=line.replace("\'s","")
    line=line.replace("\'"," ")
    line=line.replace("\"","")
    line=line.replace("&","")
    
    linewds=line.split()
    for w in stopwords:
        while w in linewds: 
            linewds.remove(w)
    
    line=" ".join(linewds)
    
    return(line)


def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr2.csv", mix=False):
    '''Implement this module to extract
        and combine text files under train_path directory into 
    imdb_tr.csv. Each text file in train_path should be stored 
    as a row in imdb_tr.csv. And imdb_tr.csv should have two 
    columns, "text" and label'''
    
    #stopwords dict
    stopw_file=open("stopwords.en.txt",'r')
    for line in stopw_file:
        line_st=line.strip()
        stopwset.add(line_st)    
        stopwset.add(string.capwords(line_st))   
    stopw_file.close()
        
    #file_out=codecs.open(outpath+name,'w','utf-8','ignore')
    file_out=codecs.open(outpath+name,'w','ISO-8859-1')
    print(",text,polarity",file=file_out)
    n=0
    sentims=["neg","pos"]
    for sentim in sentims:
        sent_index=sentims.index(sentim)
        for dirpath, dirnames, filenames in os.walk(train_path+sentim):
            #TODO : check if the codecs parameters do not remove info, e.g. on pos/10327_7
            for filename in filenames:
                n+=1
                print(filename)
                #file_in=codecs.open(train_path+sentim+"/"+filename,'r','utf-8','ignore')
                file_in=codecs.open(train_path+sentim+"/"+filename,'r','ISO-8859-1')
                line=file_in.read()
                file_in.close()
                line=remove_common_stopwords(line,stopwset)
                print("%d,\"%s\",%d" %(n,line,sent_index),file=file_out)            
    
    file_out.close()
    return
  
if __name__ == "__main__":
    
    print("importing data from files ...")
    imdb_data_preprocess(train_path)
    print("done : output file written")
    
    #get input data into panda dataframe
    df=pd.read_csv("imdb_tr2.csv", sep=",", encoding = 'ISO-8859-1')
  
    #get test data and "normalize it"   
    dftest=pd.read_csv(test_path, sep=",", encoding = 'ISO-8859-1')
    #dftest["text"]=dftest["text"].apply(remove_common_stopwords,args=( ,stopwset))
    for i in range(len(dftest)):
        line=dftest.loc[i,"text"]
        dftest.loc[i,"text"]=remove_common_stopwords(line,stopwset)
    
    '''train a SGD classifier using unigram representation,
    predict sentiments on imdb_te.csv, and write output to
    unigram.output.txt'''
    print("--SGD classifier using unigram representation--")
    
    vectorizer=CountVectorizer()
    X=vectorizer.fit_transform(df["text"])
    y=df["polarity"]

    ##SGD_parameters = {'alpha': [0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001],\
    ##                   'n_iter': [5, 10, 20, 40, 80]}
    ## -> took too long so kept only retained value for sumbmission
    
    SGD_parameters = {'alpha': [0.001, 0.0001, 0.0000001],\
                       'n_iter': [5, 20]}

    clf=SGDClassifier(loss="hinge", penalty="l1")
    clfg=GridSearchCV(clf,SGD_parameters) #does 3-fold cross validation by default
    clfg.fit(X, y)

    #print results: 
    #Res=pd.DataFrame(clfg.cv_results_)
    #print(Res)
    
    print("Best parameters : ",clfg.best_params_)
    print("Best score :",clfg.best_score_)
    #best is alpha = 0.0001 and n_iter = 5 -> score 0.82348
    
    #compute transform matrix of testing data     
    xtest=vectorizer.transform(dftest["text"])
    
    y_pred=clfg.predict(xtest)
        
    np.savetxt("unigram.output.txt",y_pred,fmt="%d",delimiter="\n")

    
    '''train a SGD classifier using bigram representation,
    predict sentiments on imdb_te.csv, and write output to
    unigram.output.txt'''
    print("--SGD classifier using bigram representation--")
 
    vectorizer2=CountVectorizer(ngram_range=(1, 2))
    X=vectorizer2.fit_transform(df["text"])
    y=df["polarity"]
    
    clf=SGDClassifier(loss="hinge", penalty="l1")
    clfg=GridSearchCV(clf,SGD_parameters) #does 3-fold cross validation by default
    clfg.fit(X, y)

    print("Best parameters : ",clfg.best_params_)
    print("Best score :",clfg.best_score_)
    #best is alpha = 1e-07 and n_iter = 20 -> score 0.84084

    #compute transform matrix of testing data     
    xtest=vectorizer2.transform(dftest["text"])
    
    y_pred=clfg.predict(xtest)

    np.savetxt("bigram.output.txt",y_pred,fmt="%d",delimiter="\n")

           
    '''train a SGD classifier using unigram representation
     with tf-idf, predict sentiments on imdb_te.csv, and write 
     output to unigram.output.txt'''
    
    print("--SGD classifier using unigram representation with tf-idf--")

    
    vectorizer=TfidfVectorizer()
    X=vectorizer.fit_transform(df["text"])
    y=df["polarity"]


    clf=SGDClassifier(loss="hinge", penalty="l1")
    clfg=GridSearchCV(clf,SGD_parameters) #does 3-fold cross validation by default
    clfg.fit(X, y)

    print("Best parameters : ",clfg.best_params_)
    print("Best score :",clfg.best_score_)
    #best is alpha = 0.001 and n_iter = 20 -> score 0.8578

    #compute transform matrix of testing data     
    xtest=vectorizer.transform(dftest["text"])
    
    y_pred=clfg.predict(xtest)
     
    np.savetxt("unigramtfidf.output.txt",y_pred,fmt="%d",delimiter="\n")

  	
    '''train a SGD classifier using bigram representation
     with tf-idf, predict sentiments on imdb_te.csv, and write 
     output to unigram.output.txt'''
    print("--SGD classifier using bigram representation with tf-idf--")

    vectorizer2=TfidfVectorizer(ngram_range=(1, 2))
    X=vectorizer2.fit_transform(df["text"])
    y=df["polarity"]

    clf=SGDClassifier(loss="hinge", penalty="l1")
    clfg=GridSearchCV(clf,SGD_parameters) #does 3-fold cross validation by default
    clfg.fit(X, y)

    print("Best parameters : ",clfg.best_params_)
    print("Best score :",clfg.best_score_)
    #best is alpha = 0.0001 and n_iter = 5 -> score 0.85208

    #compute transform matrix of testing data     
    xtest=vectorizer2.transform(dftest["text"])
    
    y_pred=clfg.predict(xtest)
    
    np.savetxt("bigramtfidf.output.txt",y_pred,fmt="%d",delimiter="\n")
