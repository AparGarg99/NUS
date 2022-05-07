# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:47:15 2019

@author: isswan
"""

#pip install nltk
#nltk.download('stopwords')
#nltk.download('punkt')

####################Data Preparation 
import pandas as pd
news=pd.read_table('r8-train-all-terms.txt',header=None,names = ["Class", "Text"])
news.head()
a = news.groupby("Class")
a.head()

news.groupby('Class').describe()

#Select a subset from the dataframe. (crude money-fx trade)
subnews=news[(news.Class=="trade")| (news.Class=='crude')|(news.Class=='money-fx') ]

subnews.groupby('Class').describe()
print(subnews.shape)

#Count the length of each document
length=subnews['Text'].apply(len)
subnews=subnews.assign(Length=length)
subnews.head()

#Plot the distribution of the document length for each category
import matplotlib.pyplot as plt
subnews.hist(column='Length',by='Class',bins=50)

plt.figure()

#####################Data preprocessing 

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
newstopwords=stopwords.words("english") + ['the','is','it','may']
newstopwords = ['the','is','it','may']
WNlemma = nltk.WordNetLemmatizer()


def pre_process(text):
    tokens = nltk.word_tokenize(text)
    tokens=[WNlemma.lemmatize(t) for t in tokens]
    tokens=[word for word in tokens if word not in newstopwords]
    text_after_process=" ".join(tokens)
    return(text_after_process)

#Apply the function on each document
subnews['Text'] = subnews['Text'].apply(pre_process)

subnews.head()

#Count the length of each document
length=subnews['Text'].apply(len)
subnews=subnews.assign(Length=length)

#####################Data Split and Create DTM
#split the data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(subnews.Text, subnews.Class, test_size=0.30, random_state=12)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()

#####################Build training pipeline using  Naïve Bayes Model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

text_clf = Pipeline([('vect', CountVectorizer()),   #Vectorizer
                     ('tfidf', TfidfTransformer()), #DTM with TFIDF
                      ('clf', MultinomialNB()),     #ML Model
                    ])

text_clf.fit(X_train,y_train ) 

##Evaluate the model
import numpy as np
from sklearn import metrics 
predicted = text_clf.predict(X_test)
print(metrics.confusion_matrix(y_test, predicted))
print("NB:",np.mean(predicted == y_test) )

#####################Build training pipeline using  Decision Tree Model
from sklearn import tree
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                      ('clf', tree.DecisionTreeClassifier())
                    ])
clf = text_clf.fit(X_train, y_train) 

predicted = clf.predict(X_test)

print(metrics.confusion_matrix(y_test, predicted))
print("DT:",np.mean(predicted == y_test) )
  
#####################Build training pipeline using  SVM Model
from sklearn import svm
from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer(use_idf=True)),
                      ('clf', svm.LinearSVC(C=1.0))
                    ])
text_clf.fit(X_train, y_train) 
    
predicted = text_clf.predict(X_test)
 
print(metrics.confusion_matrix(y_test, predicted))
print(np.mean(predicted == y_test) )
print(metrics.classification_report(y_test, predicted))
########################Prediction on new documents
docs_new = ['Crude price is dropping ', 'interest rate is increasing']
predicted = text_clf.predict(docs_new)
print(predicted)
