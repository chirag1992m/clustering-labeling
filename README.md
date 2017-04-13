# clustering-labeling
Search Engine Architecture (NYU CS Spring-17) Project

## Datasets
* [20-newsgroup](http://people.csail.mit.edu/jrennie/20Newsgroups/)
* [RCV-1 and RCV-2](https://archive.ics.uci.edu/ml/datasets/Reuters+RCV1+RCV2+Multilingual,+Multiview+Text+Categorization+Test+collection)

## Pipeline

TF-IDF --> K-Means clustering --> Candidate Labels (JSD, k-nn) --> Wiki-search --> Candidate Labels++ --> Top Labels (MI, SP) --> Evaluation (Match@K, Mean Reciprocal Rank@K)