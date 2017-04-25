'''
Course: Search Engine Architecture
Project: Cluster-Labeling

File: cluster.py
Job: Contains clustering methods
'''
import pickle
from sklearn import cluster
from scipy import sparse

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

kmeans = cluster.KMeans(n_clusters=6, verbose=1, tol=1e-10).fit(X)

pickle.dump(kmeans, open('cluster.pkl', 'wb'))
