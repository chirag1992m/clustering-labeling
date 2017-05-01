'''
Course: Search Engine Architecture
Project: Cluster-Labeling

File: cluster.py
Job: Contains clustering methods
'''
import pickle
from sklearn import cluster, preprocessing, mixture
from collections import Counter
from sklearn.feature_extraction.text import *

def kmeans_clustering(options, all_text):
	print("Running K-Means clustering...")
	X = TfidfVectorizer(max_df=0.5, max_features=10000, min_df=3, use_idf=True).fit_transform(all_text)
	kmeans = cluster.KMeans(n_clusters=options.num_clusters, verbose=1, init='k-means++').fit(X)	
	print("Label counts: ", Counter(kmeans.labels_))
	
	if options.save_intermediate:
		pickle.dump(kmeans, open('intermediate_results/cluster.pkl', 'wb'))
		pickle.dump(X, open('intermediate_results/cluster_tfidf.pkl', 'wb'))

	return X, kmeans

def gmm_clustering(optionsm, all_text):
	print("Running Gaussian Mixture Model...")
	X = TfidfVectorizer(max_df=0.5, max_features=10000, min_df=3, use_idf=True).fit_transform(all_text)
	gmm = mixture.GaussianMixture(n_components=options.num_clusters, verbose=1).fit(X)	
	print("Converged: ", gmm.converged_)
	
	if options.save_intermediate:
		pickle.dump(gmm, open('intermediate_results/cluster_gmm.pkl', 'wb'))
		pickle.dump(X, open('intermediate_results/cluster_tfidf.pkl', 'wb'))

	return X, gmm