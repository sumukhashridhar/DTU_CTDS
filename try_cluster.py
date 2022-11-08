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
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import apriori
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

data = pd.read_csv("jishichen.csv", sep=',')

data_content = data["content"]

#clean the content, remove stop words,punctuatio from the content
# and normalize each word
def clean(text):
    """
    This function takes as input a text on which several 
    NLTK algorithms will be applied in order to preprocess it
    """
    # Removing numbers
    document = re.sub(r'\d+', ' ', text)

    # Removing special characters
    document = re.sub(r"[^a-zA-Z0-9]+", ' ', document)
    tokens = word_tokenize(document)
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



# Compute DBI score
dbi = metrics.davies_bouldin_score(content_vector .toarray(), pred_labels)


# Print the DBI and Silhoutte Scores
print("DBI Score for normal: ", dbi)

#try use pca to reduce vector dimension. then apply K-means,
#after that we could use Davies-Bouldin index to evaluate
#which kind of cluster is good
svd = TruncatedSVD(n_components=50, random_state=42)
content_vector_svd = svd.fit_transform(content_vector)

model = KMeans(n_clusters=5, random_state=42)
clusters_svd= model.fit_predict(content_vector_svd )

# Compute DBI score
dbi_svd = metrics.davies_bouldin_score(content_vector_svd,clusters_svd)


# Print the DBI and Silhoutte Scores
print("DBI Score for svd: ", dbi_svd)


def word_cloud(text,wc_title,wc_file_name='wordcloud.jpeg'):
    
    # Create WordCloud 
    word_cloud = WordCloud(width = 800, height = 500, 
                           background_color ='white', 
                         
                           min_font_size = 14).generate(text) 

    # Set wordcloud figure size
    plt.figure(figsize = (8, 6)) 
    
    # Set title for word cloud
    plt.title(wc_title)
    
    # Show image
    plt.imshow(word_cloud) 

    # Remove Axis
    plt.axis("off")  

    # save word cloud
    plt.savefig(wc_file_name,bbox_inches='tight')

    # show plot
    plt.show()



df=pd.DataFrame({"text":data_content,"labels":pred_labels})

#build a dictionary to store which clusters store which content, basically the index of content.
clusters_location = {}
for i in df.labels.unique():
    new_df=df[df.labels==i]
    text="".join(new_df.text.tolist())
    for a in df.index:
        if df.loc[a].labels == i:
            if i not in clusters_location:
                clusters_location[i] = [a]
            else:
                clusters_location[i].append(a)
                
                
                      
                 
    word_cloud(text,'cluster {}'.format(i), 'cluster {}'.format(i)+'.jpeg')
    






# fail to use apriori or fpgrowth to find frequent pairs,
# reason I gues is that memory is not big enough 
"""
word_positions = {v: k for k, v in tf_idf_vect .vocabulary_.items()}
dist_words = sorted(v for k, v in word_positions.items())
for cluster_id in pred_labels:
    for k,v in clusters_location.items():
        if cluster_id == k:
            tfidf = content_vector[v]
            tfidf[tfidf > 0] = 1
            
        
    
    
    # df is a pandas sparse dataframe
    df = pd.DataFrame.sparse.from_spmatrix(tfidf, columns=dist_words)
    rule_fp = fpgrowth(df, min_support=0.3, use_colnames=True).sort_values(by='support', ascending=False)
    rule_ap = apriori(df, min_support=0.3)
    """