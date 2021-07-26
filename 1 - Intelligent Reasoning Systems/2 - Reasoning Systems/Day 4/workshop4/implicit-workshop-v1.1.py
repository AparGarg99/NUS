# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:32:56 2020
@author: issbas
"""

############################
# DEMO: movielens dataset
############################

# For this demonstration we can either assume: 
# (a) any rating is an implicit like (since the user viewed the movie they must have thought they would like it)
# (b) only ratings >=3 are likes
#
# For similar code see https://medium.com/analytics-vidhya/implementation-of-a-movies-recommender-from-implicit-feedback-6a810de173ac
#
# More example code can also be found at:
# https://nbviewer.jupyter.org/github/jmsteinw/Notebooks/blob/master/RecEngine_NB.ipynb
# https://github.com/billydh/implicit-recommender-system/blob/master/item_recommender.py

path = 'C:/Users/aparg/Desktop/IRS/Day 4/workshop2'
os.chdir(path)
trans = pd.read_csv('u_data.csv') # movielens 100K file (userids and itemids start at 1)
trans.drop('datetime',axis=1,inplace=True)
trans.columns = ['user','item','rating']
trans = trans[trans.rating >= 3]  # keep only good ratings

# convert to binary data where each rating is either 0 or alpha
# alpha is a parameter that can be adjusted by trial and error during testing
alpha = 40
trans.rating = alpha 

# create the ratings table in sparse matrix format 
# see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
# Note1: scipy sparse matrix cannot handle strings for the user name or for item names
# hence we first convert strings to (contiguous) integers (matrix indexes) and then swap the user and item names for the indexes
# Note2: when building a model using the Implicit lib, the rating matrix rows must be items and the columns users
trans,umap,imap = maptrans(trans)  
sparse_item_user = sparse.csr_matrix((trans['rating'].astype(float), (trans['item'], trans['user'])))

#Initialize the Alternating Least Squares (ALS) model.
# you can experiment with different parameters, e.g. try setting factors = 50
model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=50)

# train the model
model.fit(sparse_item_user)

# to recommend items for a target user we first create a sparse ratings matrix with users as rows and columns as items
sparse_user_item = sparse.csr_matrix((trans['rating'].astype(float), (trans['user'], trans['item'])))

# now make recommendations for a target user
targetname = 5 # this is the actual user name (as defined in the data file)
userid = umap[targetname]; userid # get the index into the sparse matrix
recommendations = model.recommend(userid, sparse_user_item);  recommendations

# to show recommendations along with the movie names
# we load the movie titles and map to item ids
titles = pd.read_csv('u_item.csv', encoding="latin-1")
titlemap = dict(zip(titles['movie id'],titles['movie name']))
for (item,score) in recommendations: print(score,titlemap[item]) 

##############################################
# set up training and test sets
##############################################

# create the test train split using the Implicit built-in
# can see the code at: https://github.com/benfred/implicit/blob/master/implicit/evaluation.pyx

trainset, testset = train_test_split(sparse_item_user, train_percentage=0.8)
    
#rebuild the implicit model on the new training set
model.fit(trainset)

# compute p@k = #actual hits over all users / # max possible hits (i.e. total #liked items within topK)
precision_at_k(model, train_user_items=trainset.T, test_user_items=testset.T, K=10)

# Mean average precision for a set of queries is the mean of the average precision@K scores for each query
# map = (sum of average precision for each test user) / # test users
mean_average_precision_at_k(model, trainset.T, testset.T, K=10)

# compute all possible metrics, try different values for K
ranking_metrics_at_k(model, trainset.T, testset.T, K=10)

######################################
# WORKSHOP: Bookcrossings (compare performance using implict versus explicit ratings)
#####################################

path = 'E:/datasets/Bookcrossings'
os.chdir(path)
trans = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding="latin-1")
trans.columns = ['user','item','rating']
trans = trans[trans.rating == 0]; # keep only implicit ratings (as binary)
alpha = 10 # can experiment with different values for this
trans.rating = alpha
trans,umap,imap = maptrans(trans)
sparse_item_user = sparse.csr_matrix((trans['rating'].astype(float), (trans['item'], trans['user'])))
sparse_item_user.count_nonzero()  # cross-check: this should be equal to the number of transactions
model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=50)

# (1) repeat the code above to: create a training/test set split, build a model and compute p@k
# what are the best results you can obtain? and with what parameters?
# try different values for alpha, and different numbers of latent features, regularisation, model-build iterations

# Note: to change alpha without reloading the data, and without changing the train/test split, you can do the below
# (Keeping the same training and test set is important when comparing model performances) 
alpha = 40 # or any new value you wish
trainset[trainset > 0] = alpha
testset[testset > 0] = alpha

# (2) we can try treating the explicit ratings as implicit ones, eg reload the data and use
trans = trans[trans.rating > 0]
# then convert to binary ratings as done in movielens above

# is there any significant difference in performance?
# if no difference then we could (arguably) use all BX data (ratings = 0  AND ratings > 0) and treat all as binary implicit likes
# but if there is a difference then we should combine the two systems using a hybrid system

# (3) we can try treating the all ratings as binary implicit ones, eg reload the data and DO NOT remove any ratings and set all ratings = 1



