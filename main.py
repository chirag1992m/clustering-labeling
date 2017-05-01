'''
Course: Search Engine Architecture
Project: Cluster-Labeling

File: main.py
Job: Coordinate between the different components to run
different experiments
'''
import argparse, os, shutil, pickle

import loadData, cluster, impTermExtraction
import wikiSearch, ngram_gen

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='20newsgroup', choices=['20newsgroup',])
parser.add_argument("--clustering", type=str, default='kmeans', choices=['kmeans', 'gmm'])
parser.add_argument("--important_terms", type=str, default='JSD', choices=['JSD','naive'])
parser.add_argument("--no_wiki_search", action="store_true")
parser.add_argument("--judge", type=str, default="PMI", choices=['PMI', 'SP',])

parser.add_argument("--num_clusters", type=int, default=5)
parser.add_argument("--num_important_words", type=int, default=10)
parser.add_argument("--top_K", type=int, default=5)
parser.add_argument("--num_wiki_results", type=int, default=10)

parser.add_argument("--clean_data", action="store_true")
parser.add_argument("--save_intermediate", action="store_true")

options = parser.parse_args()

if options.save_intermediate:
    if not os.path.exists('intermediate_results/'):
        os.mkdir('intermediate_results')

if os.path.exists('evaluations/'):
   shutil.rmtree('evaluations/')

os.mkdir('evaluations')

print("Loading and preprocessing dataset... ")
if options.dataset == '20newsgroup':
    indexes, labels, all_text = loadData.load_20_newsgroup(options)
print("Loading and preprocessing of dataset DONE!")

print("Clustering documents...")
if options.clustering == 'kmeans':
    X, cluster = cluster.kmeans_clustering(options, all_text)
elif options.clustering == 'gmm':
    X, cluster = cluster.gmm_clustering(options, all_text)
print("Clustering DONE!")

print("Extracting important terms...")
if options.important_terms == 'naive':
    weights = impTermExtraction.naive_weighing(options, X, cluster, indexes)
elif options.important_terms == 'JSD':
    weights = impTermExtraction.JSD(options, X, cluster, indexes)
important_terms = impTermExtraction.get_important_terms(options, weights['weighed_terms'])
print("extracting important terms DONE!")


if not options.no_wiki_search:
    print("Running wiki-search over the important terms...")
    wiki_labels = []
    for i in range(options.num_clusters):
        top_imp_terms = [x[0] for x in important_terms[i]]
        wiki_labels.append(wikiSearch.extractCandidateLabels(options, top_imp_terms))

    if options.save_intermediate:
        pickle.dump(wiki_labels, open('intermediate_results/wiki_labels.pkl', 'wb'))
    print("Wiki Search DONE!")

if options.judge == 'PMI':
    print("Judging terms using MI...")
    if not os.path.exists('data_supplements/ngrams_brown.pkl'):
        print("Brown N-Grams not found, generating them...")
        ngram_gen.generate_brown_ngrams()
        print("Done!")

    import topLabels
    for i in range(options.num_clusters):
        topLabels.MI(important_terms[i], wiki_labels[i], 'evaluations/topK_MI_{}.pkl'.format(i), options.top_K)
    print("Judging done")
elif options.judge == 'SP':
    print("Judging terms using SP...")
    import topLabels
    for i in range(options.num_clusters):
        topLabels.SP(wiki_labels[i], 'evaluations/topK_SP_{}.pkl'.format(i), options.top_K)
    print("Judging done")
