# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:20:08 2022

@author: geng8
"""

import sys
import os
import mmh3
import numpy as np
import itertools
import collections
import argparse
import snapy
from datasketch import MinHashLSH
import pandas as pd
from datasketch import MinHash
from datasketch import MinHashLSH
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
import pandas as pd
import os
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer
import re

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
#our data have a special punctuation '—' , need to be removed.
exclude.add('—')
exclude.add('“')
exclude.add('”')

#initialize word normalizer 
lemma = WordNetLemmatizer()



count2=1
datafolder = os.path.join('ats_corpus_small')
docs = {}
for file in os.listdir(datafolder):
    filepath = os.path.join(datafolder, file)
    f = open(filepath, 'r',encoding="utf-8")
    docs["m{}".format(count2)] = f.read()
    print("read document " + file)
    count2+=1
    f.close()
    
    
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
    tokens = [word for word in tokens if not word in stopwords.words("english")]
    # Lemmatize
    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(word, pos = "v") for word in tokens]
    tokens = [lemma.lemmatize(word, pos = "n") for word in tokens]
    return tokens

docs1={}
#initialize word normalizer 
lemma = WordNetLemmatizer()
stop = set(stopwords.words('english'))
for i,k in docs.items():
  cleaned = clean(docs[i])   
  docs1[i] = str([' '.join(cleaned)]) 
  
def hashed_lst_shingles(q, doc):
    
    doc = doc.split(" ")
    lst_shingles=[]

    lst_shingles = [doc[i:i+q] for i in range(0, len(doc), q-2)] # create list of shingles of length q

    lst_shingles = [x for x in lst_shingles if len(x)==q] # remove shingles with length < q


    #lst_shingles_h = list(set(lst_shingles)) # remove duplicates

    return lst_shingles 

docs_clean = {}
count = 1
for i,k in docs1.items():
  docs_clean["m{0}".format(count)] = hashed_lst_shingles(3,k)
  count +=1 
  
num_perm = 100
min_dict1 = {}
count3 = 1
for val in tqdm(docs_clean.values()):
    m = MinHash(num_perm=num_perm)
    for shingle in val:    
      a= []
      for i in shingle:
        i.encode('utf8')
        a.append(i)
      data1 = str(['_'.join(a)])

      m.update(data1.encode('utf8'))
    min_dict1["m{}".format(count3)] = m
    count3+=1
    
lsh = MinHashLSH(threshold=0.54, num_perm=num_perm)
for key in tqdm(min_dict1.keys()):
    lsh.insert(key,min_dict1[key]) # insert minhash data structure

def create_cand_pairs():
    big_list = []
    for query in min_dict1.keys():
        bucket = lsh.query(min_dict1[query])
        if len(bucket)==1:
            big_list.append([bucket[0],"None"])
        if len(bucket)>1:
            first_val = bucket[0]
            for val in bucket[1:]:
                second_val = val
                big_list.append([first_val,second_val])
    return big_list

a = create_cand_pairs()
b = [i for i in a if i[1] != 'None']
