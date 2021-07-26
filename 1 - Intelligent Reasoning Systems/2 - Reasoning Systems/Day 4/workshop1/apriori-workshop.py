# -*- coding: utf-8 -*-
"""
###################################################################################
Building and Testing Recommendation Systems using Association Mining: Workshop

@author: Barry Shepherd

before running this code, load and execute: apriori-lib.py and apriori-testing.py

###################################################################################
"""

import os
import numpy as np
import pandas as pd
from random import sample 
import matplotlib.pyplot as plt

############################################################################################
# (1A) Build and test association rules using kaggle groceries dataset
# this has 167 distinct grocery items, format = purchase transactions, one row per transaction
############################################################################################

path = 'C:/Users/aparg/Desktop/IRS/Day 4/workshop1'
os.chdir(path)
trans = pd.read_csv('Groceries_dataset.csv') # load in the purchase tranactions
trans.columns = ['user','datetime','item']; trans
allitems = np.unique(trans.item); len(allitems) # derive the distinct items
baskets = trans.groupby('user')['item'].apply(list); baskets  # group transactions into baskets (a series of lists)

# do some simple data visualisation/data exploration
itemfreqcnts = itemcounts(baskets) # count item frequencies
sorted(itemfreqcnts.items(), key=lambda kv: kv[1], reverse=True) # reverse sort by frequency

# display as histogram
rankeditems = [k for k,v in sorted(itemfreqcnts.items(), key=lambda kv: kv[1], reverse=True)]
frequencies = [v for k,v in sorted(itemfreqcnts.items(), key=lambda kv: kv[1], reverse=True)]
topN=20
plt.barh(rankeditems[0:topN], frequencies[0:topN], align='center', alpha=0.5)

# build a set of association rules, experiment using different support and confidence parameters
freqItemSet, rules = apriori(baskets, minSup=0.2, minConf=0.2); len(rules) # 0
freqItemSet, rules = apriori(baskets, minSup=0.1, minConf=0.1); len(rules) # 26
freqItemSet, rules = apriori(baskets, minSup=0.01, minConf=0.1); len(rules) # 9726

# examine the top rules and determine the number of unique items that the rules can recommend (target)
rules[0:10]
showrules(rules, N=50)
ruleRHSitems = RHSitems(rules); len(ruleRHSitems)
ruleRHSitems # show each unqiue targeted item and number of rules that recommend that item

# to test the rules we first divide the baskets into training and test sets and then rebuild the ruleset
testsize = int(len(baskets)*0.1); testsize # set the size of the test set
testids  = sample(list(baskets.index),testsize)
trainids = list(set(baskets.index) - set(testids))
trainbaskets = baskets[trainids]
testbaskets  = baskets[testids]

# rebuild the ruleset using the training baskest only
freqItemSet, rules = apriori(trainbaskets, minSup=0.01, minConf=0.1); len(rules) 

# make recommendations for one basket (e.g. for one user at basket checkout time)
testbasket = testbaskets.iloc[1]; testbasket
execrules_anymatch(testbasket, rules) # allows any subset of the testbasket to match a rule LHS

# make up a new basket usign any items in the inventory
testbasket = ['frozen meals','snack products']
execrules_anymatch(testbasket, rules)

# test the ruleset on the testset using holdout_1 testing ....
# we set topN = 5 and tests per basket (tpb) = 5
# are the rules better than random? check the lift over random


# A. Our rule set is nearly 8 times better than making random recommendations.
_ = rulehits_holdout_lift(testbaskets, rules, allitems, topN=5, tpb=5)


###########################################################################################
# (1B) Explore applying association mining to a web browsing session.
# This dataset records visits to the microsoft website (www.microsoft.com) for a one week period
# each record is a cookieID plus a pagecategoryID (vroot ID). 
# Predicting the next webpagecategory (vroot) a user might view can be useful for website optimisation and/or
# for recommending unseen content
############################################################################################

path = 'C:/Users/aparg/Desktop/IRS/Day 4/workshop1'
os.chdir(path)

# load the pagecategory view events
trans = pd.read_csv('anonymous-msweb-transactions.txt',sep = '\s+') 
trans.columns = ['user','item']
trans.item.nunique() # show the number of vroots

# load the category names
pagecatnames = pd.read_csv('anonymous-vrootnames-msweb.csv') 
pagecatnames.columns = ['item','title','url']
pagecatnames['title'] = pagecatnames['title'].str.strip()
pagecatnames

# join
trans = pd.merge(trans, pagecatnames, on="item")
trans

allitems = np.unique(trans.title); len(allitems) # get the distinct items

baskets = trans.groupby('user')['title'].apply(list)  # convert transactions into baskets (a series of lists)
baskets[0:30]

# proceed as above in (1A) to answer the workshop quiz questions

# Q. What is the most frequently visited vroot (website category)? 
# A. Free Downloads
itemfreqcnts = itemcounts(baskets) # count item frequencies
sorted(itemfreqcnts.items(), key=lambda kv: kv[1], reverse=True) # reverse sort by frequency

# Q. How many rules were generated? (use minSup=0.01, minConf=0.1)
# A. 467
freqItemSet, rules = apriori(baskets, minSup=0.01, minConf=0.1); len(rules) 


# Q. How many distinct Vroots are recommendable using the ruleset generated for Q2?
# A. 20
rules[0:10]
showrules(rules, N=50)
ruleRHSitems = RHSitems(rules); len(ruleRHSitems)
ruleRHSitems # show each unqiue targeted item and number of rules that recommend that item


# Q. What is the top recommendation made to a user who has just visited the following two vroots: India, Games
# A. Free Downloads
testbasket = ['India', 'Games']
execrules_anymatch(testbasket, rules)

# Calculate lift
# A. holdbacks= 7056 recitems= 20793 hits= 3818 (18.36%) randrecitems= 20793 randhits= 71 (0.34%) rulelift=53.77
testsize = int(len(baskets)*0.1); testsize # set the size of the test set
testids  = sample(list(baskets.index),testsize)
trainids = list(set(baskets.index) - set(testids))
trainbaskets = baskets[trainids]
testbaskets  = baskets[testids]

freqItemSet, rules = apriori(trainbaskets, minSup=0.01, minConf=0.1); len(rules) 

_ = rulehits_holdout_lift(testbaskets, rules, allitems, topN=3, tpb=5)


# Q. Assume that the ruleset lift you obtained in the last question was exactly one , then what would this mean?
# A. The ruleset recommendations are no better and no worse than the random recommendations.

################################################################
# (1C) Build and test associations using grocery data that also contains demographic information for the users.
# This dataset contains only has 11 grocery items. It is in tabular format (one row per user).
# We will do two tests and compare the results: 
# test1 ~ using only the grocery items to build the rules
# test2 ~ using both grocery items and user demographics (as virtual items) to build the rules
##################################################################

path = 'C:/Users/aparg/Desktop/IRS/Day 4/workshop1'
os.chdir(path)
users = pd.read_csv('baskets.txt') 
users

############# test1 ###############

# for test1 we remove the demographic & account columns - leave only the items purchased
users.drop(['cardid','value', 'pmethod', 'sex', 'homeown', 'income', 'age'], inplace=True, axis=1) # only leave the groceries
users

# before converting to baskets, we replace the value T in all of the grocery variables with the groceryname
# if we dont do this then the one-hot coded column names are assigned the values "T" and "F" 
groceryitems = list(users.columns); groceryitems
for col in groceryitems: users[col] = np.where(users[col]=='T', col, '')  # if the item was not in the basket then leave value blank
users

# convert each dataframe row into a basket (a list) and
# assemble all baskets into a series (suitable for the apriori library)

#first we define a function to do this (hence highlight all of the function below and execute)
def df2Baskets(users):
    baskets = users.values.tolist()
    newbaskets = list()
    for basket in baskets:
        while ('' in basket): basket.remove('') # remove the empty items
        if (len(basket) > 0): newbaskets.append(basket)
    return(pd.Series(newbaskets))

#now apply the function    
baskets = df2Baskets(users); baskets

# split the baskets into train and test sets
testsize = 100
testids  = sample(list(baskets.index),testsize)
trainids = list(set(baskets.index) - set(testids))
trainbaskets = baskets[trainids]
testbaskets  = baskets[testids]

# build the rules, remember to experiment with min.support and min.confidence
freqItemSet, rules = apriori(baskets, minSup=0.01, minConf=0.1); len(rules) 

# do holdout1 test
_ = rulehits_holdout_lift(testbaskets, rules, groceryitems, topN=10, tpb=5)

# Q. try different values for topN, what do the results tell you?
# A. The ruleset performed better than random for all of the values for topN but its recommendations got worse as the value for topN was increased
for n in range(1,10):
    print('For topN =',n)
    _ = rulehits_holdout_lift(testbaskets, rules, groceryitems, topN=n, tpb=10)
    print('*'*10)

########## test2 ##############

# for test2 we build and test rules that also include the card holders demographics

users = pd.read_csv('baskets.txt')  # re-read the data
users.drop(['cardid'], inplace=True, axis=1) # only drop cardID, keep all other fields
users

# the numerical fields must be converted to categorical variables for association mining
# create a function to convert a number into a category based on a set of thresholds
def tobucket(val,name,thresholds):
    prevthresh = 0
    for thresh in thresholds:
        if val <= thresh: return f'{name}{prevthresh}-{thresh}'
        prevthresh = thresh
    return f'{name}>={thresh}' 

# apply the function to the numerical variables
users['value']  = [tobucket(x,'transval:',[10,20,30,40]) for x in users['value']]
users['income'] = [tobucket(x,'income:',[10000,20000,30000]) for x in users['income']]
users['age']    = [tobucket(x,'age:',[10,20,30,40]) for x in users['age']]
users

# before converting to baskets, we replace the value T in all of the grocery variables with the groceryname
for col in groceryitems: users[col] = np.where(users[col]=='T', col, '') 
# also do same for homeown variable
users['homeown'] = np.where(users['homeown']=='YES', 'homeown', '') 
users

baskets = df2Baskets(users); baskets

# divide into training and testsets using same indexes (same trainids and testids) as used in test1 above
trainbaskets = baskets[trainids]
testbaskets  = baskets[testids]

# build a new ruleset and examine the top rules (by confidence)
# can you draw any obvious conclusions by inspecting the rules (e.g. make any easy discoveries)?
freqItemSet, rules = apriori(trainbaskets, minSup=0.05, minConf=0.1); len(rules) 
showrules(rules)

# the rules may contain non-grocery item in their RHS
# we must remove these since our goal is only to recommend groceryitems
groceryrules =list()
for LHS, RHS, conf in rules:
    intersect = set(groceryitems).intersection(RHS)
    if len(intersect) > 0: 
        groceryrules.append([LHS,intersect,conf])       
len(groceryrules)
showrules(groceryrules) # examine the top rules (by confidence)

# do holdout test, does the demographic information inprove the reommendation performance?
# (note: try running the below a few times, the lifts may vary due to the randomness of the random recommendations)
for n in range(1,10):
    _ = rulehits_holdout_lift(testbaskets, groceryrules, groceryitems, topN=n, tpb=10, itemstart=6)

   
###########################################################################################
# (1D) compare the recommendations obtained using the above association rules in 1C with the recommendations obtained
# using a decision tree classifier (or any other supervised prediction model type you choose) to build one decision model per item.
# This is also a familiarity exercise for using scikitlearn prediction models and decisiontrees
############################################################################################

from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

####################################################
# (step1) load and preprocess the data
# preproceesing includes one-hot encoding the categorical variables


users = pd.read_csv('baskets.txt') 
users
del users['cardid'] 
users.columns

# swap the value T/F in all of the grocery variables to the grocery item name (this aids the onehot coding below)
# if we dont do this then the one-hot coded column names are assigned the values "T" and "F" 
groceryitems = ['fruitveg', 'freshmeat', 'dairy', 'cannedveg', 'cannedmeat','frozenmeal', 'beer', 'wine', 'softdrink', 'fish', 'confectionery']
for col in groceryitems:
    users[col] = np.where(users[col]=='T', col, 'none') 
users['homeown'] = np.where(users['homeown']=='YES', 'homeown', 'none')  # also do same for homeown variable

# onehot encode all of the categorical variables
# after one-hot coding a variable, we delete the original (unencoded) variable
catvars = set(users.columns)-set(('value','income','age')); catvars
for v in catvars:
    onehot = pd.get_dummies(users[v])
    if 'none' in list(onehot.columns): onehot.drop('none', inplace=True, axis=1)
    users.drop(v, inplace=True, axis = 1)
    users = users.join(onehot)
 
# view the preprocessed dataset (do a visual check for correctness)
users.columns
users

####################################################
# (step2) Build and test a decision tree for one item, with no pruning

# create train and test split
testsize = int(len(users)*0.2); testsize # set the size of the test set (20%)
testnames = set(sample(list(users.index),testsize)); len(testnames) 
trainnames = set(users.index) - testnames;  len(trainnames)
train = users.loc[trainnames,]; train.shape
test = users.loc[testnames]; test.shape

# build the decision tree
target = 'fruitveg' # select the target item, any item will do
inputvars = list(set(users.columns) - set([target]))
tclf = tree.DecisionTreeClassifier()
tclf.fit(train[inputvars],train[target])
train[target]
# view the tree method1
text_representation = tree.export_text(tclf, feature_names=inputvars)
print(text_representation)
   
# view the tree method2
plt.rcParams['figure.dpi'] = 200 # adjust to get the plot resolution you want
_ = tree.plot_tree(tclf, feature_names=inputvars, class_names=target, filled=True)
  
# test the tree
preds = tclf.predict(test[inputvars])
print(classification_report(test[target],preds))
print(confusion_matrix(test[target], preds)) # rows = actual, cols = preds, eg precision for class 0 = (0,0)/((0,0)+(1,0))

# try some tree pruning to improve prediction accuracy (experiment with different pruning amounts to try to get best accuracy)
tclf = tree.DecisionTreeClassifier(min_samples_leaf=20)
tclf.fit(train[inputvars],train[target])
preds = tclf.predict(test[inputvars]); preds
predprobability = tclf.predict_proba(test[inputvars]); predprobability  # each prediction is a pair or probabilties (for class=F and class=T) 

# has the accuracy improved? if not then try a different amount of pruning or type of pruning
print(classification_report(test[target],preds))
print(confusion_matrix(test[target], preds))

####################################################
# (step3) build a tree for each grocery item and use these trees to make and test recommendations

# build the trees (one per grocery item) and store in a dictionary
trees = dict()
for target in groceryitems:
    inputvars = list(users.columns)
    inputvars.remove(target)
    clf = tree.DecisionTreeClassifier(min_samples_leaf=20)
    clf.fit(train[inputvars],train[target])
    print(target,'tree size=',clf.tree_.node_count) # size of the tree
    trees[target] = clf
    
 
# do the test for a range of topN values
# are the results better than the association rule results?
# (note: try running the below a few times, the lifts may vary due to the randomness of the random recommendations)

for n in range(1,5): 
    print("n=",n,end=" ")       
    _ = classifierhits_holdout_lift(trees, test, groceryitems, topN=n)
    
####################################################
    




    
