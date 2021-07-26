# -*- coding: utf-8 -*-
"""
Demonstrate simple collaborative filtering
@author: barry shepherd
"""

#########################################################
# simple movies dataset
#########################################################

# load the data
path = 'C:/Users/aparg/Desktop/IRS/Day 4/workshop2'
os.chdir(path)
trans = pd.read_csv('simplemovies-transactions.csv')

# convert from ratings events into a ratings matrix
# Note: uids and iids map the user and item names to matrix indexes (starting at 0)
ratmatrix, uids, iids = makeratingsmatrix(trans) 
ratmatrix.shape
ratmatrix
uids

# make recommendations for a specific user
targetname = "Toby" 
targetrats = ratmatrix[uids[targetname],] # note: user into the ratings matrix is by index, e.g. user "10" ~ index 9
targetrats

# When using pearson the recommendations for Toby should be: 3.35 (night), 2.83 (lady), 2.53 (luck)
getRecommendations_UU(targetrats, ratmatrix, iids, simfun=pearsonsim, topN = 10)

# compute item-item similarity matrix (to use the timer, select all 3 lines and then run) 
tic = time.perf_counter()
itemsims = getitemsimsmatrix(ratmatrix, simfun=euclidsimF) # use euclidsimF to agree with book/slide calcs
print(f"time {time.perf_counter() - tic:0.4f} seconds")

head(itemsims) # view the similarity matrix

# II recommendations for Toby with euclideanF shd be: 3.183 (night), 2.598 (luck), 2.473 (lady)
getRecommendations_II(targetrats, itemsims, iids, topN = 10)

# lets try pre-normalising the data
rowmeans = np.nanmean(ratmatrix,axis=1); rowmeans
normratmatrix = ratmatrix.copy()
for i in range(ratmatrix.shape[0]):  # iterate over rows
    normratmatrix[i] = normratmatrix[i] - rowmeans[i]
head(normratmatrix)

# redo the UU recommendations
targetrats = normratmatrix[uids[targetname],] 
recs = getRecommendations_UU(targetrats, normratmatrix, iids, simfun=pearsonsim, topN = 10); recs # the normalised rating predictions
recs + rowmeans[uids[targetname]] # the unnormalised rating predictions, should be similar but not idential to the unnormalised predictions


#########################################################
# TESTING the recommendations 
#########################################################

# load a bigger data so that we can split into training and test sets
trans = pd.read_csv('simplemovies-transactions-moreusers.csv'); trans
ratmatrix, uids, iids = makeratingsmatrix(trans)
ratmatrix
ratmatrix.shape

# extract a testset from the rating events by random sampling
testsize = 10
testevents = trans.sample(n=testsize).values.tolist()
testevents

# blank out the testset ratings in the ratings matrix - this now becomes our training data
for (uname,iname,rating) in testevents: 
    ratmatrix[uids[uname],iids[iname]] = np.nan 

# try using each of these in turn 
simfun = pearsonsim 
simfun = cosinesim
simfun = euclidsim

# execute the test function in demolib to make ratings predictions for the test events and obtain the prediction errors
# note: to display progress, this function prints a "." for every testevent processed
errs = computeErrs_UU(testevents, ratmatrix, uids, iids, simfun=simfun)
errs
np.nanmean(abs(errs))

# calc the item similarity matrix
# try using each of these in turn
simfun = euclidsim
simfun = cosinesim

tic = time.perf_counter()
itemsims = getitemsimsmatrix(ratmatrix, simfun = simfun)
print(f"time {time.perf_counter() - tic:0.4f} seconds")

errs = computeErrs_II(testevents, ratmatrix, uids, iids, itemsims)
np.nanmean(abs(errs))

# compute percentage rank for each test event 
# (% position of test event in the list of unseenitems ranked descending by their predicted rating, a low % position is good)
prs = computePercentageRanking(testevents, ratmatrix, uids, iids, simfun=cosinesim) #  user-based
np.nanmean(abs(prs))

prs = computePercentageRanking(testevents, ratmatrix, uids, iids, itemsims=itemsims) # item-based (since item sim.matrix is supplied)
np.nanmean(abs(prs))

# compute lift over random, where lift = (#hits using CF)/(#hits using random)
# Note: small values of topN may not yield any hits , hence below we try a range of values for topN

# lift for user-based CF
for k in [5,10,25,50,75,100,200]:
    cfhits, randhits, avgrecs, avgunseen = computeLiftOverRandom(testevents, ratmatrix, uids, iids, alg = "uu", simfun=simfun, topN=k)
    print("\ntopN=",k,"lift=", cfhits/randhits if randhits > 0 else "NA",
          "cfhits=", cfhits, "randhits=", randhits, randhits, "avgrecs=", avgrecs, "avgunseen=",avgunseen)
    
# lift for item-based CF
for k in [5,10,25,50,75,100,200]:
    cfhits, randhits, avgrecs, avgunseen = computeLiftOverRandom(testevents, ratmatrix, uids, iids, alg = "ii", itemsims=itemsims, topN=k)
    print("\ntopN=",k,"lift=", cfhits/randhits if randhits > 0 else "NA",
          "cfhits=", cfhits, "randhits=", randhits, "avgrecs=", avgrecs, "avgunseen=",avgunseen)

#########################################################
# movielens dataset
#########################################################

trans = pd.read_csv('u_data.csv') # movielens 100K file (user and itemids start at 1)
trans.drop('datetime',axis=1,inplace=True)
ratmatrix, uids, iids = makeratingsmatrix(trans)
ratmatrix.shape
sparsity(ratmatrix) # show % that is empty
ratmatrix

# select any user at random to make recommendations to, e.g.:
targetname = 10 # a movielens user
targetrats = ratmatrix[uids[targetname],] 
uurecs = getRecommendations_UU(targetrats, ratmatrix, iids, simfun=pearsonsim, topN = 20); uurecs

# to view the recommended movie names
titles = pd.read_csv('u_item.csv') # movielens 100K file (user and itemids start at 1)
for i in uurecs.index: print("rat=%2.2f, movie=%d, %s" % (uurecs['predrating'][i], i, titles['movie name'][i]))

# to make recommendations using item-based CF
itemsims = getitemsimsmatrix(ratmatrix, simfun=euclidsim) # takes ~ 20-30secs
iirecs = getRecommendations_II(targetrats, itemsims, iids, topN = 20) 
for i in iirecs.index: print("rat=%2.2f, movie=%d, %s" % (iirecs['predrating'][i], i, titles['movie name'][i]))

#----------------------
# An interesting aside:
# how many of the topN recommendations from user-based CF are also in the topN from item-based CF? 
# To compute this we convert the recommended items into sets and compute the intersection.
# Note1: a bigger value for topN is more likely to result in a bigger intersection.
# Note2: an item does not have to be in the intersection to be a good recommendation - items in the union are also potentially good recommendations
uuset = set(uurecs.index)
iiset = set(iirecs.index)
intersect = uuset.intersection(iiset)
for i in intersect: print("rat=%2.2f, movie=%d, %s" % (iirecs['predrating'][i], i, titles['movie name'][i]))
#---------------------

################################ Un-normalized #####################################

trans = pd.read_csv('u_data.csv') # movielens 100K file (user and itemids start at 1)
trans.drop('datetime',axis=1,inplace=True)
ratmatrix, uids, iids = makeratingsmatrix(trans)


testsize = 200  
testevents = trans.sample(n=testsize).values.tolist(); 
testevents

# blank out the testset ratings in the ratings matrix - this now becomes our training data
for (uname,iname,rating) in testevents: 
    ratmatrix[uids[uname],iids[iname]] = np.nan 

# try using each of these in turn 
simfun = pearsonsim 
simfun = cosinesim
simfun = euclidsim

# execute the test function in demolib to make ratings predictions for the test events and obtain the prediction errors
# note: to display progress, this function prints a "." for every testevent processed
errs = computeErrs_UU(testevents, ratmatrix, uids, iids, simfun=simfun)
errs
np.nanmean(abs(errs))

################################ Normalized #####################################

trans = pd.read_csv('u_data.csv')
trans.drop('datetime',axis=1,inplace=True)
ratmatrix, uids, iids = makeratingsmatrix(trans)

# normalize
rowmeans = np.nanmean(ratmatrix,axis=1); rowmeans
normratmatrix = ratmatrix.copy()
for i in range(ratmatrix.shape[0]):  # iterate over rows
    normratmatrix[i] = normratmatrix[i] - rowmeans[i]

testevents2 = []   
for (uname,iname,rating) in testevents: 
    l=[uname,iname,normratmatrix[uids[uname],iids[iname]]]
    testevents2.append(l)
testevents2
    
    
# blank out the testset ratings in the ratings matrix - this now becomes our training data
for (uname,iname,rating) in testevents2: 
    normratmatrix[uids[uname],iids[iname]] = np.nan 

# try using each of these in turn 
simfun = pearsonsim 
simfun = cosinesim
simfun = euclidsim

# execute the test function in demolib to make ratings predictions for the test events and obtain the prediction errors
# note: to display progress, this function prints a "." for every testevent processed
errs = computeErrs_UU(testevents2, normratmatrix, uids, iids, simfun=simfun)
errs
np.nanmean(abs(errs))

#########################################################
# Jester dataset
#########################################################

# dataset2

trans = pd.read_csv("jester_ratings.dat", sep='\s+',header=0)
ratmatrix, uids, iids = makeratingsmatrix(trans)
ratmatrix.shape
sparsity(ratmatrix) # show % that is empty
head(ratmatrix)

testsize = 10000 # for II it can be much larger

# Proceed as above to create train/test sets and to compute MAE using user-based & item-based CF
# and to explore performance of the various similarity measures.
# Note: pearsonsim may be too slow to test for this dataset but cosinesim and euclidsim should be ok
# What MAE is acceptable given that the ratings range -10->10 is much larger than movielens 1->5 ?
# Note: For testsizes > 200, I suggest commenting out the line "print('.', end = '')" in demolib.py (fn computeLiftOverRandom)


testevents = trans.sample(n=testsize).values.tolist()
testevents

# blank out the testset ratings in the ratings matrix - this now becomes our training data
for (uname,iname,rating) in testevents: 
    ratmatrix[uids[uname],iids[iname]] = np.nan 

#################################### User-based ##############################################
# calc the item similarity matrix
# try using each of these in turn
simfun = euclidsim
simfun = cosinesim

# execute the test function in demolib to make ratings predictions for the test events and obtain the prediction errors
# note: to display progress, this function prints a "." for every testevent processed
errs = computeErrs_UU(testevents, ratmatrix, uids, iids, simfun=simfun)
errs
np.nanmean(abs(errs))


# compute percentage rank for each test event 
# (% position of test event in the list of unseenitems ranked descending by their predicted rating, a low % position is good)
prs = computePercentageRanking(testevents, ratmatrix, uids, iids, simfun=simfun) #  user-based
np.nanmean(abs(prs))


# compute lift over random, where lift = (#hits using CF)/(#hits using random)
# Note: small values of topN may not yield any hits , hence below we try a range of values for topN

# lift for user-based CF
for k in [5,10,25,50,75,100,200]:
    cfhits, randhits, avgrecs, avgunseen = computeLiftOverRandom(testevents, ratmatrix, uids, iids, alg = "uu", simfun=simfun, topN=k)
    print("\ntopN=",k,"lift=", cfhits/randhits if randhits > 0 else "NA",
          "cfhits=", cfhits, "randhits=", randhits, randhits, "avgrecs=", avgrecs, "avgunseen=",avgunseen)
    
    
#################################### Item-based ##############################################
# calc the item similarity matrix
# try using each of these in turn
simfun = cosinesim
simfun = euclidsim

itemsims = getitemsimsmatrix(ratmatrix, simfun = simfun)

errs = computeErrs_II(testevents, ratmatrix, uids, iids, itemsims)
np.nanmean(abs(errs))

# compute percentage rank for each test event 
# (% position of test event in the list of unseenitems ranked descending by their predicted rating, a low % position is good)
prs = computePercentageRanking(testevents, ratmatrix, uids, iids, itemsims=itemsims) # item-based (since item sim.matrix is supplied)
np.nanmean(abs(prs))

# compute lift over random, where lift = (#hits using CF)/(#hits using random)
# Note: small values of topN may not yield any hits , hence below we try a range of values for topN
for k in [5,10,25,50,75,100,200]:
    cfhits, randhits, avgrecs, avgunseen = computeLiftOverRandom(testevents, ratmatrix, uids, iids, alg = "ii", itemsims=itemsims, topN=k)
    print("\ntopN=",k,"lift=", cfhits/randhits if randhits > 0 else "NA",
          "cfhits=", cfhits, "randhits=", randhits, "avgrecs=", avgrecs, "avgunseen=",avgunseen)



#########################################################
# book crossings dataset (optional: if time permits)
#
# This dataset is too big to fit into memory in a uncompressed (non-sparse) matrix format
# Instead we proceed by data sampling: selecting only the most popular books and the most active users
#########################################################

path = "D:/datasets/Bookcrossings"
os.chdir(path)
trans = pd.read_csv("BX-Book-Ratings.csv", sep=';', error_bad_lines=False, encoding="latin-1")
trans.columns = ['user','item','rating']
trans.shape

# remove implicit ratings
trans = trans[trans.rating != 0]
trans.shape

#trans.to_csv("BXexplicit.csv",index=False)

# this fails since the uncompressed ratings matrix is too big to fit into memory
ratmatrix, uids, iids = makeratingsmatrix(trans) 

# reduce dataset size by sampling
min_item_ratings = 10 # book popularity threshold 
popular_items = trans['item'].value_counts() >= min_item_ratings
popular_items = popular_items[popular_items].index.tolist(); len(popular_items)  # get list of popular items

min_user_ratings = 10 # user activity threshold
active_users = trans['user'].value_counts() >= min_user_ratings
active_users = active_users[active_users].index.tolist(); len(active_users) # get list of active users

print('original data: ',trans.shape)
trans = trans[(trans['item'].isin(popular_items)) & (trans['user'].isin(active_users))] # apply the filter
print('data after filtering: ', trans.shape)

# converting to a ratings matrix now succeeds
ratmatrix, uids, iids = makeratingsmatrix(trans)
ratmatrix.shape

testsize = 100 # suggested value only

# Proceed as above to create train/test sets 
# and to compute MAE using user-based & item-based CF
# and to explore performance of the various similarity measures.
# What MAE is acceptable given that the ratings range is 1 to 10?
# Note: computing the item-similarity matrix may take some time (~ 5 to 10mins) if the 
# user activity and book popularity thresholds are set at 10. 
# If you are impatient then increasing the thesholds to 20 reduces the dataset size and speeds up computation time



