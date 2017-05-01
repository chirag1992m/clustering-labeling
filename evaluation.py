import pickle
import gensim
import os
import operator

model = gensim.models.Word2Vec.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin', binary=True)

def accuracy(truth_file, pred_path, K):
    with open(truth_file) as f:
        true_labels = pickle.load(f)

    label_list = []
    for label in true_labels:
        label_list.append(label.split('.'))

    pred_labels = []
    pred_files = os.listdir(pred_path)
    for pfile in pred_files:
        with open(pfile) as f:
            pred_labels.append(pickle.load(f))

    for pred_list in pred_labels:
        for true_list in label_list:
            clust_list = []
            clust_sum = 0
            preds = pred_list.split()
            for pred_label in preds:
                for true_label in true_list:
                    clust_sum += model.wv.similarity(true_label, pred_label)
            clust_list.append(([pred_list, true_list], clust_sum))
        clust_sum_sorted = sorted(clust_list, key=operator.itemgetter(1), reverse=True)
        print(clust_sum_sorted[:K])
