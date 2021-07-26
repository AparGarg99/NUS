# -*- coding: utf-8 -*-
"""
Executing and testing assocation rules
Barry Shepherd (Feb, 2021)

"""

import numpy as np
from random import sample

# Holdback-1 test for a set of association rules on a testset, also includes lift over random.
# We assume virtual items (if any) occur at the start of the basket:
# usually virtual items (eg age, gender) are not items that we wish to recommend hence testitemstart indicates 
# the start of the items that are to be tested (ie can be recommended).
# For each basket: do tpb (testsperbasket) tests by holding out in turn the first, second, third etc items in the testitems.
# Compute a random recommendation only when a rule-based recommendation is also made (for accurate comparison with the ruleset).
def rulehits_holdout_lift(testbaskets, rules, allitems, topN=10, tpb=5, itemstart=0):
    tothits = tottests = totrecs = totrhits = totrrecs = 0
    for testbasket in testbaskets:
        virtualitems = testbasket[:itemstart]
        testitems   = testbasket[itemstart:]
        numtests = min(len(testitems),tpb)
        for i in range(0,numtests):
            recs = execrules_anymatch(virtualitems+testitems[:i]+testitems[i+1:], rules, topN) # omit (holdout) the ith testitem
            nrecs = len(recs)
            if (nrecs > 0):
                recitems = set()
                for item, conf in recs: recitems.add(item) # strip out the confidences
                tothits = tothits + int(testitems[i] in recitems) # increment if testitem is in the recommended items
                totrecs = totrecs + nrecs
                tottests = tottests + 1
                # now do the random recommendations
                unseenitems = set(allitems) - set(testitems[:i]+testitems[i+1:]) # remove the holdout item
                recitems = sample(unseenitems,min(topN,len(unseenitems),nrecs))
                nrecs = len(recitems)
                totrhits = totrhits + int(testitems[i] in recitems) # increment if testitem is in the recommended items
                totrrecs = totrrecs + nrecs
    if (totrecs == 0 or totrrecs == 0 or totrhits == 0):
        print("no recommendations made, please check your inputs")
        return np.nan
    print("#holdbacks=",tottests,
          "recitems=",totrecs,
          "hits=",tothits,
          "({:.2f}%)".format(tothits*100/totrecs),
          "randrecitems=",totrrecs,
          "randhits=",totrhits,
          "({:.2f}%)".format(totrhits*100/totrrecs),
          "rulelift={:.2f}".format((tothits/totrecs)/(totrhits/totrrecs))) 
    return tothits, totrecs, tottests, totrhits, totrrecs

# to facilitate holdout sets > 1
# do split basket test for the rules (each testbasket = trainitemsubset + testitemsubset)
# the rules are executed using trainitemsubset for the LHS and the rule RHS is then compared with testitemsubset
def rulehits_splitbaskets(testbaskets, rules, topN=10):
    tothits = totrecs = 0
    uniqueitems = set()
    for trainbasket, testbasket in testbaskets:
        recs = execrules_anymatch(trainbasket, rules, topN)
        nrecs = len(recs)
        if (nrecs > 0):
            recitems = set()
            for item, conf in recs: recitems.add(item) # strip out the confidences
            tothits = tothits + len(recitems.intersection(testbasket))
            totrecs = totrecs + min(nrecs,len(testbasket))  # nrecs
            uniqueitems = uniqueitems.union(recitems)
    return tothits, totrecs, len(uniqueitems)

# do split basket test using random recommnendations
def randomhits_splitbaskets(testbaskets, allitemset, topN=10):
    tothits = totrecs = 0
    for trainbasket, testbasket in testbaskets:
        unseenitems = allitemset - set(trainbasket) # remove the seen (training) items
        recitems = set(sample(unseenitems,min(topN,len(unseenitems)))) # make topN random recommendations
        nrecs = len(recitems)
        if (nrecs > 0):
            tothits = tothits + len(recitems.intersection(testbasket))
            totrecs = totrecs + min(nrecs,len(testbasket))     # do we only count the size of the testset? (as max #recs possible) 
    return tothits, totrecs


# make and test recommendations from a set of classifiers (one per item) for the test users
def classifierhits_holdout_lift(classifiers,testset,allitems,topN=1):
    hits = randhits = tests = 0
    allpreds = list()
    for username in testset.index:
        user = testset.loc[[username]] # extract user as dataframe
        for testitem in allitems:
            if (user.loc[username,testitem] == 1): # its been bought by user hence can use as a holdback (test) item
                tests += 1
                probs = dict()
                unseenitems = list()
                user.loc[username,testitem] = 0 # blank out value
                # make a prediction (exec the corresponding tree) for every item not yet seen/bought by user
                for unseenitem in allitems: 
                    if (user.loc[username,unseenitem] == 0):  # its a valid unseen item
                        unseenitems.append(unseenitem)
                        inputvars = list(user.columns)
                        inputvars.remove(unseenitem)
                        pred = classifiers[unseenitem].predict_proba(user[inputvars])
                        probs[unseenitem] = pred[0][1] # get prob for class = True (the second element), Note: can check order returned with clf.classes_
                user.loc[username,testitem] = 1 # restore holdback value
                recs = sorted(probs.items(), key=lambda kv: kv[1], reverse=True) # sort unseen items by reverse probability
                numrecs = min(len(recs),topN)
                for item, conf in recs[0:numrecs]:
                    if (item == testitem):
                        hits += 1; break
                if (testitem in sample(unseenitems,numrecs)): randhits += 1 # make random recommedations           
                allpreds.append((testitem,recs[0:numrecs])) # record the recommendations made
    lift = hits/randhits if randhits > 0 else np.nan
    print("tests=",tests,"rulehits=",hits,"randhits=",randhits,"lift=", lift)
    return tests, hits, randhits, allpreds
    
#######################################################
# association rule execution
#######################################################

def execrules_allbaskets(baskets,rules,topN=10):
    recs = list()
    for basket in baskets: 
        rec = execrules_anymatch(basket,rules,topN)
        print(basket,"->",rec,"\n")
        recs.append(execrules_anymatch(basket,rules,topN))
    return (recs)

# execute a rule if any subset of the basket matches a rule LHS
# does not return any RHS item if its also within the LHS
# if many rules output (recommend) the same item then return the highest confidence for that item
# outputs a list of tuples: (item, confidence) for the topN items with highest confidence
def execrules_anymatch(itemset,rules,topN=10):
    preds = dict()
    for LHS, RHS, conf in rules:
        if LHS.issubset(itemset):
            for pitem in RHS:
                # ignore rules like A => A
                if not pitem in itemset:
                    if pitem in preds.keys():
                        preds[pitem] = max(preds[pitem],conf)
                    else:
                        preds[pitem] = conf                
    recs = sorted(preds.items(), key=lambda kv: kv[1], reverse=True)
    return recs[0:min(len(recs),topN)]

# only execute a rule if all of the basket matches the rule LHS
def execrules_exactmatch(itemset,rules,topN=10):
    preds = dict()
    for LHS, RHS, conf in rules:
        if LHS == set(itemset):
            for pitem in RHS:
                # ignore rules like A => A
                if not pitem in itemset:
                    if pitem in preds.keys():
                        preds[pitem] = max(preds[pitem],conf)
                    else:
                        preds[pitem] = conf                
    recs = sorted(preds.items(), key=lambda kv: kv[1], reverse=True)
    return recs[0:min(len(recs),topN)]

# return the set of all items in the ruleset RHS's
def RHSitems(rules):
    allitems = dict()
    for LHS, RHS, conf in rules:
        for item in RHS:
            if item in allitems.keys():
                allitems[item] = allitems[item] + 1
            else:
                allitems[item] = 1
    return allitems

# return the unique set of all items (as freq dictionary)
def itemcounts(itemsets):
    allitems = dict()
    for its in itemsets:
        for i in its:
            if i in allitems.keys():
                allitems[i] = allitems[i] + 1
            else:
                allitems[i] = 1  
    return allitems

def itemhist(idict):
    return sorted(idict.items(), key=lambda kv: kv[1], reverse=True)

# nice rule print (prints in reverse: confidence, RHS, KHS)
def showrules(rules, N=30):
    for L, R, C in rules: 
        print("{:.2f}".format(C),"\t",R,"<=\t",L)
        N = N-1
        if N <= 0: break
    
def clearall():
    get_ipython().magic('reset -sf')  # to clear all data


        


        
        
        
    
