import wikipedia
'''
TODO: Only the first title of the term was taken into consideration.
All titles used currently.
All categories of the page are added.
'''


def extractLabels(input_file, output_file):
    imp_terms = []
    with open(input_file) as f:
        for line in f:
            imp_terms.append(line.strip())
    f = open(output_file, "wb")
    for term in imp_terms:
        new_terms = []
        doc_list = wikipedia.search(term)
        if doc_list == []:
            continue
        for doc in doc_list:
            try:
                doc_page = wikipedia.page(doc)
                title = doc_page.title
                categories = doc_page.categories
                new_terms.append(title)
                new_terms.extend(categories)
                for nterm in new_terms:
                    f.write(nterm + "\n")
                # f.write("\n")
            # check this error
            except wikipedia.exceptions.DisambiguationError:
                pass
    f.close()
