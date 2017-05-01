import math
import pickle
import operator

ngrams = pickle.load(open('data_supplements/ngrams_brown.pkl', 'rb'))

def getCount(word):
    gram = len(word.split())
    if word in ngrams[gram]:
        return ngrams[gram][word] / ngrams[str(gram)+"_count"]
    else:
        return 1


def MI(terms, termlist, output_file, K):
    imp_terms = [term[0] for term in terms]
    scores = [term[1] for term in terms]

    scores = [float(i) / max(scores) for i in scores]

    mi = {}

    for terms in termlist:
        for label in terms:
            mi[label[0]] = 0
            for idx, term in enumerate(imp_terms):
                search_query = label[0] + " " + term
                combined_count = getCount(search_query)
                label_count = getCount(label[0])
                term_count = getCount(term)
                pmi = math.log(((combined_count + 1)/ label_count) / (term_count + 2), 10)
                mi[label[0]] += (pmi * scores[idx])

    mi = sorted(mi.items(), key=operator.itemgetter(1))
    topk = mi[:K]
    labels = [label[0] for label in topk]

    with open(output_file, "wb") as f:
        pickle.dump(labels, f)


def SP(candt_input, output_file, K):
    with open(candt_input, 'rb') as f:
        cand_terms = pickle.load(f)

    label_set = set()
    for term_list in cand_terms:
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

    sp = sorted(sp.items(), key=operator.itemgetter(1), reverse=True)
    topk = sp[:K]
    labels = [label[0] for label in topk]
    with open(output_file, "wb") as f:
        pickle.dump(labels, f)
