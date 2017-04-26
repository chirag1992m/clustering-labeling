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

indexes = pickle.load(open('indexes.pkl', 'rb'))

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

X = sparse.coo_matrix((data, (rows, columns)), shape=(doc_count, vocab_count))
X = preprocessing.normalize(X, norm='l2', copy=False)

kmeans = cluster.KMeans(n_clusters=7, verbose=1, random_state=42).fit(X)

print(Counter(kmeans.labels_))

pickle.dump(kmeans, open('cluster.pkl', 'wb'))
