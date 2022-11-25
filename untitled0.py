import sys
import os
import mmh3
import numpy as np
import itertools
import collections
import pandas as pd
import numpy as np
#################### Utilities ######################
#hashes a list of strings
def listhash(l,seed):
	val = 0
	for e in l:
		val = val ^ mmh3.hash(e, seed)
	return val 

################### Similarity ######################

docs = {} #dictionary mapping document id to document contents

data = pd.read_csv(r"C:\Users\hasee\Documents\comptools\DTU_CTDS\data\articles2.csv")

#%%

data1 = data[:2000]
key = data1["Unnamed: 0"]
data2 = data1["content"]
docs = data2.to_dict()

#%%

''' Returns document as a list of hashes'''
'''
Creates shingles of size q, removing shingles of size < q. Removes duplicates and hashes the result.
'''
def hashed_lst_shingles(q, doc):
    
    doc = doc.split(" ")
    lst_shingles=[]

    lst_shingles = [doc[i:i+q] for i in range(0, len(doc), q-2)] # create list of shingles of length q

    lst_shingles = [x for x in lst_shingles if len(x)==q] # remove shingles with length < q

    seed=0
    lst_shingles_h = [listhash(shingle, seed) for shingle in lst_shingles] # hash the singles

    lst_shingles_h = list(set(lst_shingles_h)) # remove duplicates

    return lst_shingles_h

# doc = "You and me, we made a vow. For better or for worse. I can't believe you let me down"
# lst_shnigles = hashed_lst_shingles(3, doc)
# print(lst_shnigles)

#%%

def generate_matrices(docs, shingle_size=3, num_hash_funcs=2):
    
    doc_hashes = [hashed_lst_shingles(shingle_size, docs[key]) for key in docs.keys()] # returns list of lists, where each elem contains hash list for that document.
    len(doc_hashes)
    my_elements = list(itertools.chain(*doc_hashes)) # flatten to get a single list
    my_elements = list(set(my_elements)) # set() to remove duplicates. This gives the universe.

    ''' Iterate over documents (S1, S2, etc.).
        insert 1 if element from universe exists in the doc else 0
        finally, concatenate all lists to get matrix m
    '''
    boolean_m = []

    for my_doc in doc_hashes:
        my_doc=set(my_doc) # Without this, the code is too slow to run!
        # my_doc.shape
        boolean_m.append([1 if element in my_doc else 0 for element in my_elements])

    m = np.array(boolean_m).T # Matrix m!
   
    ''' Compute hashes to get matrix h_m'''
    rows = np.arange(len(my_elements))
    
    hashes_lst = []
    for i in range(num_hash_funcs):
        hashes_lst.append([mmh3.hash(x, seed=i) for x in rows])

    h_m = np.array(hashes_lst).T # Matrix h_m!
    

    return m, h_m

def min_hash_alg(m, h_m):
    sig_m = np.full((h_m.shape[1], m.shape[1]), np.inf) # Signature matrix. This stores the final result!

    my_args = np.argwhere(m == 1)
    for args in my_args:
        row=args[0]
        column=args[1]
        for i in range(h_m.shape[1]):
            if h_m[row][i] < sig_m[i, column]:
                sig_m[i, column] = h_m[row][i]
    
    return sig_m

#%%

m, h_m = generate_matrices(docs, shingle_size=3, num_hash_funcs=100)
m.shape
sig_m = min_hash_alg(m, h_m)
# sig_m

#%%

''' Jaccard similarity'''
def jaccard(s1, s2):
    return len(s1 & s2) / len(s1 | s2)

score = jaccard(set(sig_m[:, 5]), set(sig_m[:, 6])) # finding similarity b/w documents 5 and 6. remember00palm.txt and remembermeorholy00palm.txt

print(score)

#%%

''' Implementation of LSH, dividing signature matrix into b band with r rows each'''
def LSH(b, r):
    b=20
    r=5
    #b*r=num_hash_funcs

    sim_hashes=[]
    start=0
    for i in range(b):
        sim_hashes.append([listhash(col, seed=i) for col in sig_m[start:start+r,:].T])
        start=i+r

    return sim_hashes

''' Find candidate pairs by checking to see if the hashes match.
Then we check to see that the Jaccard similarity b/w each pair of docs is atleast t. If so, we consider it a candidate pair otherwise not '''
def get_cand_pairs(sim_hashes, t):
    cand_pairs=set()
    for L in sim_hashes:
        dups = collections.defaultdict(list)
        for i, e in enumerate(L):
            dups[e].append(i)
        for k, v in sorted(dups.items()):
            if len(v) >= 2:
                cand_pairs.add(tuple(v))
                # print(k, v)
    cand_pairs=list(cand_pairs)
    filtered_cand_pairs = [pair for pair in cand_pairs if (jaccard(set(sig_m[:, pair[0]]), set(sig_m[:, pair[1]])) > t)]
    return filtered_cand_pairs
    # return cand_pairs
    
#%%

b=20
r=5
sim_hashes = LSH(b=20, r=5)
pairs=get_cand_pairs(sim_hashes, t=(1/b)**(1/r))
pairs






