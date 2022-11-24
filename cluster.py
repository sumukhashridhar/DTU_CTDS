# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 20:27:23 2022

@author: geng8
"""

import pandas as pd
import os
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer
import re
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt 
from collections import defaultdict
from efficient_apriori import apriori
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics


#preprocess our data
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
#our data have a special punctuation '—' , need to be removed.
exclude.add('—')
exclude.add('“')
exclude.add('”')

#initialize word normalizer 
lemma = WordNetLemmatizer()

data = pd.read_csv("articles1.csv", sep=',')

data_content = data["content"]

#clean the content, remove stop words,punctuatio from the content
# and normalize each word
def clean(text):
    """
    This function takes as input a text on which several 
    NLTK algorithms will be applied in order to preprocess it
    """
  
    tokens = word_tokenize(text)
    # Remove the punctuations
    tokens = [word for word in tokens if word not in stop and len(word) > 2]
    # Lower the tokens
    tokens = [word.lower() for word in tokens]
    # Remove stopword
    tokens = [word for word in tokens if not word in stop]
    
    return tokens



#replace the original content with cleaned content 
for i in data_content.index:  
    cleaned = clean(data_content[i])   
    data_content.loc[i] = str([' '.join(cleaned)])
  
    


#Count Vectoriser then tidf transformer
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data_content)
from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X)

num_clusters = 3
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf)
print(km.labels_)
clusters = km.labels_.tolist()