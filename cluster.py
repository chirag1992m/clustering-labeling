'''
Course: Search Engine Architecture
Project: Cluster-Labeling

File: cluster.py
Job: Contains clustering methods
'''
import pickle, os
from sklearn import cluster, preprocessing, mixture
from collections import Counter
from sklearn.feature_extraction.text import *

def kmeans_clustering(options, all_text):
	print("Running K-Means clustering...")
	X = TfidfVectorizer(max_df=0.5, max_features=10000, min_df=3, stop_words='english', use_idf=True).fit_transform(all_text)
	kmeans = cluster.KMeans(n_clusters=options.num_clusters, verbose=1, init='k-means++').fit(X)	
	print("Label counts: ", Counter(kmeans.labels_))
	
	if options.save_intermediate:
		pickle.dump(kmeans, open(os.path.join(options.intermediate_out_directory, 'cluster.pkl'), 'wb'))
		pickle.dump(X, open(os.path.join(options.intermediate_out_directory, 'cluster_tfidf.pkl'), 'wb'))

	return X, kmeans

#Too slow for large input
def gmm_clustering(options, all_text):
	print("Running Gaussian Mixture Model...")
	X = TfidfVectorizer(max_df=0.5, max_features=10000, min_df=3, stop_words='english', use_idf=True).fit_transform(all_text)
	gmm = mixture.GaussianMixture(n_components=options.num_clusters, verbose=1).fit(X.toarray())	
	print("Converged: ", gmm.converged_)
	
	if options.save_intermediate:
		pickle.dump(gmm, open(os.path.join(options.intermediate_out_directory, 'cluster_gmm.pkl'), 'wb'))
		pickle.dump(X, open(os.path.join(options.intermediate_out_directory, 'cluster_tfidf.pkl'), 'wb'))

	return X, gmm

def birch_clustering(options, all_text):
	print("Running Birch Clustering...")
	X = TfidfVectorizer(max_df=0.5, max_features=10000, min_df=3, stop_words='english', use_idf=True).fit_transform(all_text)
	c = cluster.Birch(n_clusters=options.num_clusters).fit(X)	
	print("Label counts: ", Counter(c.labels_))
	
	if options.save_intermediate:
		pickle.dump(c, open(os.path.join(options.intermediate_out_directory, 'cluster_birch.pkl'), 'wb'))
		pickle.dump(X, open(os.path.join(options.intermediate_out_directory, 'cluster_tfidf.pkl'), 'wb'))

	return X, c

def ac_clustering(options, all_text):
	print("Running Agglomerative Clustering...")
	X = TfidfVectorizer(max_df=0.5, max_features=10000, min_df=3, stop_words='english', use_idf=True).fit_transform(all_text)
	c = cluster.AgglomerativeClustering(n_clusters=options.num_clusters).fit(X.toarray())	
	print("Label counts: ", Counter(c.labels_))
	
	if options.save_intermediate:
		pickle.dump(c, open(os.path.join(options.intermediate_out_directory, 'cluster_ac.pkl'), 'wb'))
		pickle.dump(X, open(os.path.join(options.intermediate_out_directory, 'cluster_tfidf.pkl'), 'wb'))

	return X, c
