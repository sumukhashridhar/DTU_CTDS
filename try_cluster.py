# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 22:27:17 2022

@author: geng8
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 18:49:27 2022

@author: geng8
"""

import pm4py as pm
import pandas as pd
import os
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer
import re
from sklearn.cluster import KMeans

stop = set(stopwords.words('english'))

exclude = set(string.punctuation)
#our data have a special punctuation '—' , need to be removed.
exclude.add('—')

#initialize word normalizer 
lemma = WordNetLemmatizer()

data = pd.read_csv("articals1.csv", sep=',')

data_content = data["content"]



#clean the content, remove stop words,punctuatio from the content
# and normalize each word
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    processed = re.sub(r"\d+","",normalized)
    y = processed.split()
    return y

#replace the original content with cleaned content 
for i in data_content.index:  
    cleaned = clean(data_content[i])   
    data_content.loc[i] = str([' '.join(cleaned)])
  
    


# Initialize regex tokenizer
tokenizer = RegexpTokenizer(r'\w+')

# # Vectorize document using TF-IDF
tf_idf_vect = TfidfVectorizer(
                        ngram_range = (1,1),
                        tokenizer = tokenizer.tokenize)

# transform the each news'content into vector
content_vector = tf_idf_vect.fit_transform(data_content)



# create Kmeans clusters and fit it to the content, we can change n_clusters to have more or less clusters.
clusters = KMeans(n_clusters=5).fit(content_vector )

#visualize the cluster
pred_labels = clusters.labels_