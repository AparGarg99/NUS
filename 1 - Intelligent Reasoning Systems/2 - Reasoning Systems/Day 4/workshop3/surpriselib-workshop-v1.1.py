# -*- coding: utf-8 -*-

#from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
#from surprise.model_selection import GridSearchCV

import os
import numpy as np
import pandas as pd


"""
##################################################################
# Demo 3A: Using Surprise with User-Based and Item-Based Collaborative filtering
##################################################################
"""

########################
# load the simple movies dataset
########################

path = 'C:/Users/aparg/Desktop/IRS/Day 4/workshop2'
os.chdir(path)
trans = pd.read_csv('simplemovies-transactions.csv')
trans.columns = ['user','item','rating']

 # convert to surprise format
reader = Reader(rating_scale=(1,5)) # assumes datafile contains: user, item, ratings (in this order)
data = Dataset.load_from_df(trans, reader)
################################################################
# build a model using user-based or item-based CF
################################################################

trainset = data.build_full_trainset()  # use all data (ie no train/test split)
# select the model type, below are some examples, you can adjust the parmeters if you wish

algo = BaselineOnly() # computes baselines for all users and items
algo = KNNBasic() # default method = User-based CF, default similarity is MSD (euclidean), default k=40
algo = KNNBasic(k=40,sim_options={'name': 'pearson'}) # User-based CF using pearson
algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': False}) # item-based CF using cosine
algo = KNNWithMeans(k=40,sim_options={'name': 'pearson'}) 

algo.fit(trainset) # build the model

################################################################
# predict the rating for a specific user and item
################################################################

# select a target user
rawuid = 'Toby' 

# select an item (pick any one of the below)
rawiid = 'SnakesOnPlane' # was rated by Toby
rawiid = 'NightListener' # was not rated by Toby
rawiid = 'LadyinWater' # was not rated by Toby
rawiid = 'JustMyLuck' # was not rated by Toby

# convert user and items names (raw ids) into indexes (inner ids)
# surprise jargon: raw ids are the user & item names as given in the datafile, they can be ints or strings
# inner ids are indexes into the sorted rawids
uid = trainset.to_inner_uid(rawuid); uid
iid = trainset.to_inner_iid(rawiid); iid

# if the actual rating is known it can be passed as an argument
realrating = dict(trainset.ur[uid])[iid]; realrating  # retrieve the real rating
pred = algo.predict(rawuid, rawiid, r_ui = realrating, verbose = True)

# if the actual rating is unknown then it can be omitted
pred = algo.predict(rawuid, rawiid); pred 

# FYI: can compare with prediction made using demolib (the library used in workshop2) - results should be very close
usersA, umap, imap = makeratingsmatrix(trans)
targetuser = usersA[umap[rawuid],]; targetuser
predictrating_UU(targetuser,usersA,imap[rawiid],simfun=pearsonsim)

################################################################
# make recommendations for the target 
# to do this we must predict the ratings for all their unseen items
##################################################################

# get all of the unseen items for all users
unseen = trainset.build_anti_testset(); len(unseen) # get all ratings that are not in the trainset
unseen[1:20] # the rating shown is the global mean rating, the actual rating is unknown

# this makes predictions for all users for all of their unseen items - but it may be slow for big datasets
predictions = algo.test(unseen) ; len(predictions)
predictions

# to predict only the ratings for the target user (specfied earlier by rawuid)
# we extract the targetuser from unseen
targetonly = list()
for ruid, riid, r in unseen:
    if (ruid == rawuid):
        targetonly.append((ruid, riid, r))        
targetonly 
      
predictions = algo.test(targetonly)
predictions

# now get the actual recommended items - the topN rated items
# (note this function is defined below)
get_top_n(predictions) 

################################################################
# function to get the topN recommendations for each user
# by ranking the unseen items by their predicted rating 
# input is the rating predictions
# output is a dictionary where keys are (raw) userids and 
# values are lists of tuples: [(raw item id, pred.rating),...] 
# see https://surprise.readthedocs.io/en/stable/FAQ.html

from collections import defaultdict

def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))  
    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True) # sort on predicted rating
        top_n[uid] = user_ratings[:n]
    return top_n

###########################################################################
# To demonstrate testing using a train/test split we load the movielens data
# FYI: this datset is also preloaded into the surprise package, 
# hence we could use below line instead of reading from file
# data = Dataset.load_builtin('ml-100k') 
###########################################################################

trans = pd.read_csv('u_data.csv') # movielens 100K file
trans = trans.iloc[:,0:3] # keep only first 3 columns
trans.columns = ['user','item','rating']
trans
 # convert to surprise format
reader = Reader(rating_scale=(1,5)) 
data = Dataset.load_from_df(trans, reader)

# split data into training and test sets using surprise fn: train_test_split())
# Note: if test_size parameter is float then it represents proportion of the data, if integer it represents absolute number 
trainset, testset  = train_test_split(data, test_size=0.1); len(testset)  # select 10% of rating events (10% of 100K ~ 10K)

# show stats about the split
print('users,items in trainset=',trainset.n_users, trainset.n_items)
testdf = pd.DataFrame(testset)
print('users,items in testset=',len(testdf.iloc[:,0].unique()),len(testdf.iloc[:,1].unique()))

# the argument k is the number of neighbours to take into consideration (default is 40)
algo = KNNBasic(k=50,sim_options={'name': 'pearson', 'user_based': True}) # User-based CF using pearson
algo.fit(trainset)
preds = algo.test(testset)
accuracy.rmse(preds)
accuracy.mae(preds)

# run 5-fold cross-validation.
res = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# ########################################################################
# ASIDE: for comparison, we can use demolib on the exact same testset
# ########################################################################

# convert the rating events into a ratings matrix
rats, umap, imap = makeratingsmatrix(trans)

 # blank out the test events in the ratings matrix
for (u,i,r) in testset:  rats[umap[u],imap[i]] = np.nan

# test user-based CF using demolib
# Note: for movielens we reduce the size of testset to (say) 500, since demolib is slow for user-based CF
simfun = pearsonsim
errs = computeErrs_UU(testset[1:500], rats, umap, imap, simfun=simfun) 
np.nanmean(abs(errs))

# repeat the surprise test on the smaller testset (for a more accurate, apples-to-apples, comparison)
preds = algo.test(testset[1:500]) 
accuracy.mae(preds)

# compare Surprise versus demolib using item-based CF
algo = KNNBasic(k=40,sim_options={'name': 'MSD', 'user_based': False}) 
algo.fit(trainset)
preds = algo.test(testset)
accuracy.mae(preds)

simfun = euclidsim
itemsims = getitemsimsmatrix(rats, simfun=simfun)
errs = computeErrs_II(testset, rats, umap, imap, itemsims) # demolib item-based CF is fast, so can use the whole testset
np.nanmean(abs(errs))

####################################################
# can compute PRECISION@K and RECALL@K for a given K and rating threshold
# copy the code for fn: precision_recall_at_k() from: https://surprise.readthedocs.io/en/stable/FAQ.html
# then paste the code into the console window

precisions, recalls = precision_recall_at_k(preds, k=5, threshold=4) 
# average over all tests
print("avg P@K=", sum(prec for prec in precisions.values()) / len(precisions))
print("avg recall=",sum(rec for rec in recalls.values()) / len(recalls))


#########################################################
# Workshop 3A: 
# 
# Use the above code to load the Jester dataset (Jester dataset2)
# and compare Surprise with demolib using Item-Based CF, with similarity = Euclidean (MSD)
# Use MAE as the comparison metric
# Try altering the number of neighbours (k) used in the Surprise model
#########################################################

trans = pd.read_csv("jester_ratings.dat", sep='\s+',header=0)
trans.columns = ['user','item','rating']

reader = Reader(rating_scale=(-10,+10)) 
data = Dataset.load_from_df(trans, reader)

trainset, testset  = train_test_split(data, test_size=0.1); len(testset)

######################### DEMOLIB ##########################
rats, umap, imap = makeratingsmatrix(trans)

 # blank out the test events in the ratings matrix
for (u,i,r) in testset:  rats[umap[u],imap[i]] = np.nan

simfun = euclidsim
itemsims = getitemsimsmatrix(rats, simfun=simfun)

errs = computeErrs_II(testset, rats, umap, imap, itemsims) # demolib item-based CF is fast, so can use the whole testset
np.nanmean(abs(errs)) # 3.408

######################### SURPRISE ##########################
# compare Surprise versus demolib using item-based CF
    
algo = KNNBasic(k=40,sim_options={'name': 'MSD', 'user_based': False}) 
algo.fit(trainset)
preds = algo.test(testset)
accuracy.mae(preds)
# 3.21

"""
####################################################################################################
# Demo3B: building a model using Matrix Factorisation
#
# Note: if you wish to see the SVD (funks alg) code then you can go to...
# https://github.com/NicolasHug/Surprise/blob/master/surprise/prediction_algorithms/matrix_factorization.pyx
#
#####################################################################################################
"""

# load movelens dataset

trans = pd.read_csv('u_data.csv') # movielens 100K file
trans = trans.iloc[:,0:3] # keep only first 3 columns
trans.columns = ['user','item','rating']

reader = Reader(rating_scale=(1,5)) 
data = Dataset.load_from_df(trans, reader)

# split data into training and test sets
trainset, testset  = train_test_split(data, test_size=0.2); len(testset)  # select 10% of rating events (10% of 100K ~ 10K)

# select an MF algorithm
algo = SVD(n_factors = 50) # simon funks algorithm, default is 100 factors
algo = SVDpp(n_factors = 50) # an extension of SVD that handles implicit ratings
algo = NMF(n_factors = 50) # non negative matrix factorisation

algo.fit(trainset) # build the model

# pick a target user to make recommendations for
rawuid = 3 

# get a list of all unseen items for that user, to do this we
# get a list of all unseen items for all users, then extract only the target user
unseen = trainset.build_anti_testset() 

targetonly = list()
for ruid, riid, r in unseen:  
    if (ruid == rawuid):
        targetonly.append((ruid, riid, r))

len(targetonly) # the number of unseen items for the target user (if this is zero then go back and pick another target user)
   
# make predictions and recommendations for the target   
predictions = algo.test(targetonly)
recs = get_top_n(predictions, n=10)
recs

# to show the recommendations along with the movie names
# we first load the movie titles and create a dict to map movie id to title
titles = pd.read_csv('u_item.csv', encoding="latin-1")
titlemap = dict(zip(titles['movie id'],titles['movie name'])) 

# now show the recommendations using the  movie names
for user,rlist in recs.items(): 
    for iid, rat in rlist:
        print(rat, iid, titlemap[iid])
        
# compute MAE for the testset
preds = algo.test(testset)
accuracy.mae(preds)


#######################################################
# An Aside:
# to help understand how predictions are made when using matrix factorisation we can 
# compute the prediction ourselves from the factorised matrices and the biases: pu,qi,bu,bi

# examine top-left part of the User and Item preference matrix
algo.pu[0:10,0:10] 
algo.qi[0:10,0:10]

# examine the learned biases (these are useful for cold-start)
algo.bu[0:10] # user mean ratings
algo.bi[0:10]# item mean ratings
algo.default_prediction() # the global mean rating

# examine the data for the target user & target item
# can pick any item, but pick one of the items recommended above for easy comparison
rawuid = 3  # pick same user as above
rawiid = 408 # pick one of the items that was recommended above
uid = trainset.to_inner_uid(rawuid); uid
iid = trainset.to_inner_iid(rawiid); iid

algo.pu[uid,] # target user preferences
algo.qi[iid,]  # target item preferences
algo.bu[uid] # target user bias
algo.bi[iid] # target item bias

# manually compute the prediction, this should agree with the output from algo.predict()
pred = algo.bu[uid] + algo.bi[iid] + sum(algo.pu[uid,] * algo.qi[iid,]) + algo.default_prediction(); pred


###############################################################################################
# Workshop 3B: 
#
# Try to find the best value for n_factors (the number of latent features) when using SVD algorithm
# Do this for Movielens data and also book crossings (BX) data
# For BX, use all explict book ratings (no need to subsample)
###############################################################################################

trans = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding="latin-1")
trans.columns = ['user','item','rating']

# For BX remove implicit ratings (a rating of 0 means the book was read but not rated))
trans = trans[trans.rating != 0]
trans.shape

# select the correct ratings scale 
reader = Reader(rating_scale=(1,10)) # for BX

data = Dataset.load_from_df(trans, reader)
trainset, testset  = train_test_split(data, test_size=0.2); len(testset)

# explore different number of latent factor using the below (adjust the factor list if required)
for f in [10,20,30,40,50,60,70,80,90,100,200,500]:
    algo = SVD(n_factors = f)
    algo.fit(trainset)
    preds = algo.test(testset)
    print(f)
    accuracy.mae(preds)



