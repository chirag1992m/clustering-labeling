import requests
from bs4 import BeautifulSoup
import math
import pickle
import operator
import inventory


def getCount(word):
    r = requests.get('http://www.google.com/search', params={'q':'"' + word + '"', "tbs":"li:1"})
    soup = BeautifulSoup(r.text)
    return int(soup.find('div',{'id':'resultStats'}).text)


def MI(impt_input, candt_input, output_file):
    with open(impt_input) as f:
        terms = pickle.load(f)

    imp_terms = [term[0] for term in terms]
    scores = [term[1] for term in terms]

    scores = [float(i) / max(scores) for i in scores]

    mi = {}
    with open(candt_input) as f:
        with open(candt_input) as f:
            terms = pickle.load(f)
        for label in terms:
            mi[label] = 0
            for term in imp_terms:
                search_query = label + " " + term
                combined_count = getCount(search_query)
                label_count = getCount(label)
                term_count = getCount(term)
                pmi = math.log((combined_count / label_count) / term_count, 10)
                mi[label] += (pmi * scores[term])

    mi = sorted(mi.items(), key=operator.itemgetter(1))
    topk = mi[:inventory.K]
    labels = [label[0] for label in topk]

    with open(output_file, "wb") as f:
        pickle.dump(labels, f)


def SP(candt_input, output_file):
    with open(candt_input) as f:
        cand_terms = pickle.load(f)

    label_set = set()
    for term_list in candt_input:
        for term in term_list:
            label_set.add(term[0])

    ind_list = {}
    sp = {}
    for term in label_set:
        ind_list[term] = []
        sp[term] = 0

    total_docs = []
    scores = []
    for index, term_list in enumerate(cand_terms):
        total_docs.append(len(term_list))
        scores.append(term_list[0][1])
        for term in term_list:
            ind_list[term[0]].append(index)


    w = {}
    for term in label_set:
        w[term] = 0
        for ind in ind_list[term]:
            w[term] += (scores[ind] / total_docs[ind])

    kw = {}
    for term in label_set:
        kwords = term.split()
        for kword in kwords:
            if kword in kw:
                kw[kword] += w[term]
            else:
                kw[kword] = 0

    for term in label_set:
        kwords = term.split()
        for kword in kwords:
            sp[term] += kw[kword]

    sp = sorted(sp.items(), key=operator.itemgetter(1))
    topk = sp[:inventroy.K]
    labels = [label[0] for label in topk]
    with open(output_file, "wb") as f:
        pickle.dump(labels, f)

