# %%
import sys
import os
import mmh3
import numpy as np
import itertools
import collections
from efficient_apriori import apriori
import pandas as pd
import re
import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("words")
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.tokenize import word_tokenize

#################### Utilities ######################
#hashes a list of strings
def listhash(l,seed):
	val = 0
	for e in l:
		val = val ^ mmh3.hash(e, seed)
	return val 

################### Similarity ######################

df1 = pd.read_csv(r"C:\Users\hasee\Documents\comptools\project\data\articles1.csv") #TODO replace with sys path

df1=df1["content"]
# df_lst=df1.values.tolist()

# transactions = [tuple(row.split()) for row in df1.values.tolist()]
# transactions

# %%
# "also one" in df1[1]
# df1[1]

# %%
#clean the content, remove stop words,punctuatio from the content
# and normalize each word
def clean(text):
    stop = set(stopwords.words('english'))
    # words_set = set(words.words())
    """
    This function takes as input a text on which several 
    NLTK algorithms will be applied in order to preprocess it
    """
    # Removing numbers
    # print("1 start")
    document = re.sub(r'\d+', ' ', text)
    # print("2 start")
    # Removing special characters
    document = re.sub(r"[^a-zA-Z0-9]+", ' ', document)
    # print("3 start")
    tokens = word_tokenize(document)
    # Remove the punctuations
    # print("4 start")
    tokens = [word for word in tokens if word not in stop and len(word) > 2]
    # Lower the tokens
    # print("5 start")
    tokens = [word.lower() for word in tokens]
    # Remove stopword
    # print("6 start")
    tokens = [word for word in tokens if not word in stop]
    # remove common  words
    # tokens = [word for word in tokens if not word in words_set]
    # Lemmatize
    # lemma = WordNetLemmatizer()
    # tokens = [lemma.lemmatize(word, pos = "v") for word in tokens]
    # tokens = [lemma.lemmatize(word, pos = "n") for word in tokens]
    return tuple(tokens)



# %%
# import multiprocessing
# try:
#     cpus = multiprocessing.cpu_count()
#     print(cpus)
# except NotImplementedError:
#     cpus = 2   # arbitrary default

# def square(n):
#     return n * n

# pool = multiprocessing.Pool(processes=cpus)
# ret = pool.map(square, range(1000))

# %%
# transactions = [clean(article) for article in df1.values.tolist()[:5000]]
transactions = [tuple(row.split()) for row in df1.values.tolist()]


# %%
itemsets, rules = apriori(transactions[:1], min_support=0.8, min_confidence=0.8, output_transaction_ids=True)

# %%
len(itemsets)
# itemsets[1]


