# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:16:43 2019

@author: isswan
"""

# In this workshop we perform topic modeling using gensim
# The "Class" labels here are only used for sanity check of the topics discovered later.
# Remember, in actual use of topic modelling, the documents DON'T come with labeled classes.
# It's unsupervised learning.

import numpy as np
import pandas as pd
news=pd.read_table('MovieStories.utf8.txt',header=0)


#########################################################Preprocessing
import nltk
from nltk.corpus import stopwords
mystopwords=stopwords.words("english") + ['life','film','movie','one','two']
WNlemma = nltk.WordNetLemmatizer()

def pre_process(text):
    tokens = nltk.word_tokenize(text)
    tokens=[ WNlemma.lemmatize(t.lower()) for t in tokens]
    tokens=[ t for t in tokens if t not in mystopwords]
    tokens = [ t for t in tokens if len(t) >= 3 ]
    return(tokens)


text = news['Storyline']
toks = text.apply(pre_process)

# Use dictionary (built from corpus) to prepare a DTM (using frequency)
import logging
#pip install gensim
import gensim 
from gensim import corpora

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Filter off any words with document frequency less than 2, or appearing in more than 90% documents
dictionary = corpora.Dictionary(toks)
print(dictionary)
dictionary.filter_extremes(no_below=2, no_above=0.8)
"""
        no_below : int, optional
            Keep tokens which are contained in at least `no_below` documents.
        no_above : float, optional
            Keep tokens which are contained in no more than `no_above` documents
            (fraction of total corpus size, not an absolute number).
"""
print(dictionary)

#dtm here is a list of lists, which is exactly a matrix
dtm = [dictionary.doc2bow(d) for d in toks]
###########################################################
lda = gensim.models.ldamodel.LdaModel(dtm, num_topics = 5, id2word = dictionary, passes=10,chunksize=128,random_state=10)

lda.show_topics(10)

##Evaluate the coherence score of LDA models
'''
u_mass:prefer the model close to 0 
c_v: [0,1], prefer bigger value   
Do not fully rely on the coherence score
'''
from gensim.models.coherencemodel import CoherenceModel
cm_umass = CoherenceModel(lda,  dictionary=dictionary, corpus=dtm, coherence='u_mass')
cm_cv = CoherenceModel(lda,  dictionary=dictionary, texts=toks, coherence='c_v')
lda_umass = cm_umass.get_coherence()
lda_cv = cm_cv.get_coherence()
print(lda_umass)
print(lda_cv)


#pip install pyLDAvis
import pyLDAvis.gensim
import pickle 
import pyLDAvis
# Visualize the topics
pyLDAvis.enable_notebook()
LDAvis_prepared = pyLDAvis.gensim.prepare(lda, dtm, dictionary)
pyLDAvis.show(LDAvis_prepared)

dict = {0: 'Love', 1: 'school', 2: 'young_crime', 3: 'politic', 4:'war'}

##Note that different runs result in different but simillar results if random_state is not specified
##Label the topics based on representing "topic_words"

# Get the topic distribution of documents
doc_topics = lda.get_document_topics(dtm)

from operator import itemgetter
#show the topic distributions for the first 5 docs, 
for i in range(0, 5):
    print(doc_topics[i])
    print(max(doc_topics[i], key=itemgetter(1))[0]) 

#Select the best topic (with highest score) for each document
top_topic = [ max(t, key=itemgetter(1))[0] for t in doc_topics ]
print (top_topic)


#topics_perDoc = [ dict[t] for t in top_topic ]
#print (topics_perDoc)

####################################### How many dos in each topic?
labels, counts = np.unique(topics_perDoc, return_counts=True)
print (labels)
print (counts)

###########################Save and load pre-trained model
from gensim.test.utils import datapath
# Save model to disk.
temp_file = datapath("LDA_model")
lda.save(temp_file)
# Load a potentially pretrained model from disk.
lda = gensim.models.ldamodel.LdaModel.load(temp_file)
########################Query, the model using new, unseen documents
other_texts = [
    ['To','save','her','ailing','father','from','serving','in','the','Imperial','Army,','a','fearless','young','woman','disguises','herself','as','a','man','to','battle','northern','invaders','in','China'],#doc 1
    ['A','secret','agent','embarks','on','a','dangerous,','time-bending','mission','to','prevent','the','start','of','World','War','III'],#doc 2
]

new_dtm = [dictionary.doc2bow(d) for d in other_texts]

for i in range(0, 2):
    unseen_doc = new_dtm[i]
    print(lda[unseen_doc])


########################Update the model by incrementally training on the new corpus
lda.update(new_dtm)


for i in range(0, 2):
    unseen_doc = new_dtm[i]
    print(lda[unseen_doc])

