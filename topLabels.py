import requests
from bs4 import BeautifulSoup
import math
import pickle
import operator

K=5

def getCount(word):
    r = requests.get('http://www.google.com/search', params={'q':'"' + word + '"', "tbs":"li:1"})
    soup = BeautifulSoup(r.text)
    return int(soup.find('div',{'id':'resultStats'}).text)


def MI(impt_input, candt_input, output_file):
    with open(impt_input) as f:
        imp_terms = pickle.load(f)
        scores = pickle.load(f)

    scores = [float(i) / max(scores) for i in scores]

    mi = {}
    with open(candt_input) as f:
        for line in f:
            label = line.strip()
            mi[label] = 0
            for term in imp_terms:
                search_query = label + " " + term
                combined_count = getCount(search_query)
                label_count = getCount(label)
                term_count = getCount(term)
                pmi = math.log((combined_count / label_count) / term_count, 10)
                mi[label] += (pmi * scores[term])

    mi = sorted(mi.items(), key=operator.itemgetter(1))
    topk = mi[:K]
    labels = [label[0] for label in topk]

    with open(output_file, "wb") as f:
        pickle.dump(labels, f)


def SP(candt_input, output_file):
    with open(candt_input) as f:
        cand_terms = pickle.load(f)
        scores = pickle.loda(f)

    label_set = set.union(*map(set, cand_terms))
    ind_list = {}
    for term in label_set:
        ind_list[term] = []

    total_docs = []
    for index, term_list in enumerate(cand_terms):
        total_docs.append(len(term_list))
        for term in term_list:
            ind_list[term].append(index)

    w = {}
    for term in label_set:
        w[term] = 0
        for ind in ind_list[term]:
            w[term] += (scores[ind] / total_docs[ind])

