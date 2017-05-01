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
from sklearn.feature_extraction.text import *

def kmeans_clustering(options, all_text):
	print("Running K-Means clustering...")
	X = TfidfVectorizer(max_df=0.5, max_features=10000, min_df=3, use_idf=True).fit_transform(all_text)
	kmeans = cluster.KMeans(n_clusters=options.num_clusters, verbose=0, init='k-means++').fit(X)	
	print("Label counts: ", Counter(kmeans.labels_))
	
	if options.save_intermediate:
		pickle.dump(kmeans, open('intermediate_results/cluster.pkl', 'wb'))
		pickle.dump(X, open('intermediate_results/cluster_tfidf.pkl', 'wb'))

	return X, kmeans
