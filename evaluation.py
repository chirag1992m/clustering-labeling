import pickle
import gensim
import os
import operator

model = gensim.models.KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin', binary=True)

def accuracy(truth_file, pred_labels, K):
    with open(truth_file, 'rb') as f:
        true_labels = pickle.load(f)

    label_list = []
    for label in true_labels:
        label_list.append(label.split('.'))

    for pred_list in pred_labels:
        clust_list = []
        for true_list in label_list:
            for pred_string in pred_list:
                clust_sum = 0
                pred_strings = pred_string.split()
                for pred_label in pred_strings:
                    for true_label in true_list:
                        try:
                            clust_sum += model.similarity(true_label, pred_label)
                        except KeyError:
                            pass
                clust_list.append(([pred_string, '.'.join(true_list)], clust_sum))
        clust_sum_sorted = sorted(clust_list, key=operator.itemgetter(1), reverse=True)
        print(clust_sum_sorted[:K])

if __name__ == "__main__":
    accuracy('intermediate_results/unique_labels.pkl', 'evaluations/', 5)
