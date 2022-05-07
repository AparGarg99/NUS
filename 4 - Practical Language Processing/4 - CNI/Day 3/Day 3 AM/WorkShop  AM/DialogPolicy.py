# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 13:28:57 2019

@author: isswan
"""

from __future__ import unicode_literals
import spacy
import numpy as np
print (spacy.__version__) 
import pandas as pd  

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn

##prepare dataset
from os import getcwd, chdir
fpath = getcwd()
print (fpath)
# Change your path here
chdir(fpath) 


file_location = fpath+"//data//Dialogs_Pizza.txt"
f = open(file_location, "r")

dialogList=[]

for line in f:
    if '#' not in line:
        dialogList.append(line.split(','))
print (len(dialogList))

df = pd.DataFrame(dialogList, columns =['Prev', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9','Pred'])
X= df[['Prev', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']]
y= df['Pred']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

########################################################

clf_ME = LogisticRegression().fit(X_train,y_train)
y_pred = clf_ME.predict(X_test)

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))


model_svm = SVC(C=5000.0, gamma="auto", kernel='rbf')
clr_svm = model_svm.fit(X_train, y_train)   
y_pred = clr_svm.predict(X_test)

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))


confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
sn.heatmap(confusion_matrix, annot=True)


