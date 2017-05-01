'''
Course: Search Engine Architecture
Project: Cluster-Labeling

File: candidateLabelExtraction.py
Job: Extracting representative labels of clusters
'''
import pickle, math
from collections import Counter

def naive_weighing(options, cluster, indexes):
	term_cluster_weights = {}
	term_collection_weights = {}

	cluster_counts = Counter(cluster.labels_)

	for term in indexes['idf'].keys():
		term_collection_weights[term] = {}
		idf = indexes['idf'][term]
		tf = sum([indexes['tf'][term][t] for t in indexes['tf'][term].keys()])

		term_cluster_weights[term] = {}
		for cluster_id in range(options.num_clusters):
			tf_cluster = [indexes['tf'][term][t] for t in indexes['tf'][term].keys() if cluster.labels_[t] == cluster_id]
			ctf = sum(tf_cluster) / float(cluster_counts[cluster_id])
			cdf = math.log(1 + sum([1 for i in tf_cluster]))

			term_cluster_weights[term][cluster_id] = cdf * ctf * idf

		term_collection_weights[term] = sum([term_cluster_weights[term][t] for t in term_cluster_weights[term].keys()])

	if options.save_intermediate:
		pickle.dump({'weighed_terms': term_cluster_weights, 'collection_weights': term_collection_weights}, 
				open('intermediate_results/term_weights_naive.pkl', 'wb'))

	return {'weighed_terms': term_cluster_weights, 'collection_weights': term_collection_weights}


def JSD(options, cluster, indexes):
	naive_weights = naive_weighing(options, cluster, indexes)
	cluster_weights = naive_weights['weighed_terms']
	collection_weights = naive_weights['collection_weights']

	JSD = {}
	for term in cluster_weights.keys():
		JSD[term] = {}
		for cluster_id in range(options.num_clusters):
			P = cluster_weights[term][cluster_id]
			Q = collection_weights[term]

			M = 0.5 * (P + Q)
			if M > 0:
				D_p_m = P * math.log(1 + (P / M))
				D_q_m = Q * math.log(1 + (P / M))

				JSD[term][cluster_id] = 0.5 * (D_p_m + D_q_m)
			else:
				JSD[term][cluster_id] = 0

	if options.save_intermediate:
		pickle.dump({'weighed_terms': JSD}, 
				open('intermediate_results/term_weights_JSD.pkl', 'wb'))

	return {'weighed_terms': JSD}

def get_important_terms(options, weighed_terms):
	important_words = [None] * options.num_clusters
	for i in range(options.num_clusters):
		weights = [(term, weighed_terms[term][i]) for term in weighed_terms.keys()]
		weights.sort(key=lambda x: x[1], reverse=True)
		important_words[i] = weights[:options.num_important_words]

	if options.save_intermediate:
		pickle.dump(important_words, open('intermediate_results/important_words.pkl', 'wb'))

	return important_words
