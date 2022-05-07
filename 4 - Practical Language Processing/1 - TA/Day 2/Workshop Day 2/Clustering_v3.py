# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:04:54 2019

@author: isswan
"""
# In this workshop we perform document clustering using sklearn

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Remember, in actual use of document clustering, the documents DON'T come with labeled classes.
# It's unsupervised learning.
import numpy as np
import pandas as pd


news=pd.read_table('r8-train-all-terms.txt',header=None,names = ["Class", "Text"])
news.groupby('Class').size()

subnews=news[(news.Class=="interest")| (news.Class=='crude')|(news.Class=='money-fx') ]
subnews.head()
subnews.groupby('Class').size()


###################################### Preprocessing

import nltk
from nltk.corpus import stopwords
mystopwords=stopwords.words("english") + ['one', 'become', 'get', 'make', 'take']

WNlemma = nltk.WordNetLemmatizer()

def pre_process(text):
    tokens = nltk.word_tokenize(text)
    tokens=[ WNlemma.lemmatize(t.lower()) for t in tokens]
    tokens=[ t for t in tokens if t not in mystopwords]
    tokens = [ t for t in tokens if len(t) >= 3 ]
    text_after_process=" ".join(tokens)
    return(text_after_process)

# Apply preprocessing to every document in the training set.
text = subnews['Text']
toks = text.apply(pre_process)

####################################### TDM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

# Create tfidf matrix
vectorizer = TfidfVectorizer(max_df=0.7, max_features=2500,
                             min_df=3, stop_words=mystopwords,
                             use_idf=True)
X = vectorizer.fit_transform(toks)
X.shape


####################################### Apply KMeans for clustering
from sklearn.cluster import KMeans
from sklearn import metrics


#‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
#Maximum number of iterations of the k-means algorithm for a single run.

km3 = KMeans(n_clusters=3, init='k-means++', max_iter=2000,random_state=5)
km3.fit(X)

# Evaluate the 3 clusters 
# Coefficient: more similar within clusters, more distant between clusters
# The higher the better (-1 to 1)

print("Coefficient for 3 clusters: %0.3f"
      % metrics.silhouette_score(X, km3.labels_))

####################################### How many dos in each cluster
labels, counts = np.unique(km3.labels_[km3.labels_>=0], return_counts=True)
print (labels)
print (counts)

subnews.groupby('Class').size()

######################################### What are the clusters about
# note: Clustering only gives you index of cluster rather than the meaning of cluster
# need to review the docs in each cluster and summarize 
# We still need to see the more representative words for each cluster to understand them.

def print_terms(cm, num):
    original_space_centroids = cm.cluster_centers_
    order_centroids = original_space_centroids.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(num):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()


print_terms(km3, 3)


# Let's assign the cluster label to the categories 
# Note: the order of clusters may change for differnt runs 

dict = {0: 'interest', 1: 'crude', 2: 'money-fx'}
print(dict)
print(counts)
subnews.groupby('Class').size()

######################################### Evaluation: Assume we have the annotations 
#get the category for each document
cluster_labels = [ dict[c] for c in km3.labels_]
correct_labels = subnews['Class']
print(metrics.confusion_matrix(cluster_labels, correct_labels))
print(np.mean(cluster_labels == correct_labels) )
print(metrics.classification_report(cluster_labels, correct_labels))

########################################
####################################### Use SVD to reduce dimensions
svd = TruncatedSVD(300)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X_lsa = lsa.fit_transform(X)



#set to False to perform inplace row normalization
# Check how much "variance is explained" (information is kept)
#explained_variance = svd.explained_variance_ratio_.sum()
#print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

####################################### Apply KMeans for clustering
from sklearn.cluster import KMeans

km3 = KMeans(n_clusters=3, init='k-means++', max_iter=1000, n_init=1)
km3.fit(X_lsa)

from sklearn import metrics


print("Coefficient for 3 clusters: %0.3f" % metrics.silhouette_score(X_lsa, km3.labels_))


labels, counts = np.unique(km3.labels_[km3.labels_>=0], return_counts=True)
print (labels)
print (counts)

subnews.groupby('Class').size()

def print_SVD_terms(cm, num):
    original_space_centroids = svd.inverse_transform(cm.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(num):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()

print_SVD_terms(km3, 3)

# Let's assign the cluster label to the categories 
# Note: the order of clusters may change for differnt runs 
dict = {0: 'money-fx', 1: 'crude', 2: 'interest'}
cluster_labels = [ dict[c] for c in km3.labels_]

####Let's check out the confusion matrix of clustering results
##### Remember, in actual use of document clustering, the documents DON'T come with labeled classes.
##### So nomally we can not access the confusion matrix unless we label some data manually 
import numpy as np
print(metrics.confusion_matrix(cluster_labels, correct_labels))
print(np.mean(cluster_labels == correct_labels) )
print(metrics.classification_report(cluster_labels, correct_labels))

