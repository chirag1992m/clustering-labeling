'''
Course: Search Engine Architecture
Project: Cluster-Labeling

File: candidateLabelExtraction.py
Job: Extracting representative labels of clusters
'''
import pickle, math
from collections import Counter

cluster = pickle.load(open('cluster.pkl', 'rb'))
indexes = pickle.load(open('indexes.pkl', 'rb'))

term_cluster_weights = {}
term_collection_weights = {}

cluster_counts = Counter(cluster.labels_)


for term in indexes['idf'].keys():
	term_collection_weights[term] = {}
	idf = indexes['idf'][term]
	tf = sum([indexes['tf'][term][t] for t in indexes['tf'][term].keys()])

	for cluster_id in range(6):
		tf_cluster = [indexes['tf'][term][t] for t in indexes['tf'][term].keys() if cluster.labels_[t] == cluster_id]
		ctf = sum(tf_cluster) / float(cluster_counts[cluster_id])
		cdf = math.log(1 + sum([1 for i in tf_cluster]))

		term_cluster_weights[term][cluster_id] = cdf * ctf * idf

	term_collection_weights[term] = sum([term_cluster_weights[term][t] for t in term_cluster_weights[term].keys()])

JSD = {}

'''
JSD(P || Q) = 1/2 * D(P || M) + 1/2 * D(Q || M)
M = 1/2(P + Q)
D(X || Y) = X(t) * log(1 + X(t) / Y(t))
'''

for term in term_cluster_weights.keys():
	JSD[term] = {}
	for cluster_id in range(6):
		P = term_cluster_weights[term][cluster_id]
		Q = term_collection_weights[term]

		M = 0.5 * (p + q)
		if M > 0:
			D_p_m = P * math.log(1 + (P / M))
			D_q_m = Q * math.log(1 + (P / M))

			JSD[term][cluster_id] = 0.5 * (D_p_m + D_q_m)
		else:
			JSD[term][cluster_id] = 0


pickle.dump({'cluster_weights': term_cluster_weights, 'collection_weights': term_collection_weights, 'JSD': JSD}, 
	open('term_weights', 'wb'))

