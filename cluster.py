'''
Course: Search Engine Architecture
Project: Cluster-Labeling

File: cluster.py
Job: Contains clustering methods
'''
import pickle
from sklearn import cluster, preprocessing
from scipy import sparse
from collections import Counter
import inventory
from sklearn.feature_extraction.text import *
# from sklearn.decoposition import *
from sklearn.pipeline import *

indexes = pickle.load(open('indexes.pkl', 'rb'))
all_text = pickle.load(open('all_text.pkl', 'rb'))

doc_count = indexes['doc_count']
vocab_count = len(indexes['idf'].keys())

rows = []
columns = []
data = []
for idx, word in enumerate(indexes['tf'].keys()):
    idf = indexes['idf'][word]
    for doc in indexes['tf'][word]:
        rows.append(doc)
        columns.append(idx)
        data.append(indexes['tf'][word][doc] * idf)

# X = sparse.coo_matrix((data, (rows, columns)), shape=(doc_count, vocab_count))
# X = preprocessing.normalize(X, norm='l2', copy=False)

X = TfidfVectorizer(max_df=0.5, max_features=10000, min_df=2, stop_words='english', use_idf=True).fit_transform(all_text)

kmeans = cluster.KMeans(n_clusters=inventory.NUM_CLUSTERS, verbose=1, init='k-means++').fit(X)
print(Counter(kmeans.labels_))

pickle.dump(kmeans, open('cluster.pkl', 'wb'))
