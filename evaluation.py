import pickle
import gensim
import os
import operator
import argparse

model = gensim.models.KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin', binary=True)

def accuracy(true_labels, pred_labels):
    score = 0

    for tl in true_labels:
        max_score = 0
        for pred_label in pred_labels:
            pred_label = pred_label.split()
            for pl in pred_label:
                try:
                    current_score = model.similarity(tl, pl)
                    if current_score > max_score:
                        max_score = current_score
                except KeyError:
                    continue
        score += max_score

    return score

if __name__ == "__main__":
    true_labels = ["Computers", "Sports", "Politics", "Religion", "miscellaneous"]
    parser = argparse.ArgumentParser()
    parser.add_argument('in_directory', 
        help="Directory containing evaluated labels", 
        default="evaluations", type=str)
    options = parser.parse_args()

    pred_labels = []
    for f in os.listdir(options.in_directory):
        df = os.path.join(options.in_directory, f)
        pred_labels.extend([x.strip() for x in open(df, 'r').readlines()])

    print("Accuracy: {}".format(accuracy(true_labels, pred_labels)))