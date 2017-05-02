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
parser.add_argument("-d", "--dataset", type=str, default='20newsgroup', choices=['20newsgroup'], help="Dataset to use.")
parser.add_argument("-c", "--clustering", type=str, default='kmeans', choices=['kmeans', 'gmm', 'birch', 'ac'], help="Clustering algorithm to be used")
parser.add_argument("-i", "--important_terms", type=str, default='JSD', choices=['JSD','naive'], help="Algorithm for choosing the cluster representative terms")
parser.add_argument("-nw", "--no_wiki_search", action="store_true", help="Label clusters without the searching over wikipedia")
parser.add_argument("-j", "--judge", type=str, default="PMI", choices=['PMI', 'SP',], help="The judging mechanism to be used")

parser.add_argument("-nc", "--num_clusters", type=int, default=5, help="Number of clusters to be made from dataset")
parser.add_argument("-ni", "--num_important_words", type=int, default=20, help="Number of important words to be extracted")
parser.add_argument("-K", "--top_K", type=int, default=20, help="K of the Top-K results")
parser.add_argument("--num_wiki_results", type=int, default=15, help="Number of top wiki results to be used")

parser.add_argument("-clean", "--clean_data", action="store_true", help="Delete downloaded data after usage")
parser.add_argument("-s", "--save_intermediate", action="store_true", help="Save any intermediate results during the whole process")
parser.add_argument("-o", "--out_directory", type=str, default="evaluations", help="Directory where the final names will be displayed")
parser.add_argument("-io", "--intermediate_out_directory", type=str, default="intermediate_results", help="Directory where all the intermediate files will be saved")

options = parser.parse_args()

if options.save_intermediate:
    if not os.path.exists(options.intermediate_out_directory):
        os.mkdir(options.intermediate_out_directory)

if os.path.exists(options.out_directory):
   shutil.rmtree(options.out_directory)

os.mkdir(options.out_directory)

print("Loading and preprocessing dataset... ")
if options.dataset == '20newsgroup':
    indexes, labels, all_text = loadData.load_20_newsgroup(options)

print("Clustering documents...")
if options.clustering == 'kmeans':
    X, cluster = cluster.kmeans_clustering(options, all_text)
elif options.clustering == 'gmm':
    X, cluster = cluster.gmm_clustering(options, all_text)
elif options.clustering == 'birch':
    X, cluster = cluster.birch_clustering(options, all_text)
elif options.clustering == 'ac':
    X, cluster = cluster.ac_clustering(options, all_text)

print("Extracting important terms...")
if options.important_terms == 'naive':
    weights = impTermExtraction.naive_weighing(options, X, cluster, indexes)
elif options.important_terms == 'JSD':
    weights = impTermExtraction.JSD(options, X, cluster, indexes)
important_terms = impTermExtraction.get_important_terms(options, weights['weighed_terms'])

if not options.no_wiki_search:
    print("Running wiki-search over the important terms...")
    wiki_labels = []
    for i in range(options.num_clusters):
        top_imp_terms = [x[0] for x in important_terms[i]]
        wiki_labels.append(wikiSearch.extractCandidateLabels(options, top_imp_terms))

    if options.save_intermediate:
        pickle.dump(wiki_labels, open(os.path.join(options.intermediate_out_directory, 'wiki_labels.pkl'), 'wb'))
    print("Wiki Search DONE!")

if options.judge == 'PMI':
    print("Judging terms using MI...")
    if not os.path.exists('data_supplements/ngrams_brown.pkl'):
        print("Brown N-Grams not found, generating them...")
        ngram_gen.generate_brown_ngrams()

    import topLabels
    for i in range(options.num_clusters):
        topLabels.MI(important_terms[i], wiki_labels[i], os.path.join(options.out_directory, 'topK_MI_{}.txt'.format(i)), options.top_K)
elif options.judge == 'SP':
    print("Judging terms using SP...")
    import topLabels
    for i in range(options.num_clusters):
        topLabels.SP(wiki_labels[i], os.path.join(options.out_directory, 'topK_SP_{}.txt'.format(i)), options.top_K)


print("Done! The extracted labels are available in the directory: ", options.out_directory)