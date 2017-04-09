import os
import tarfile
import pickle
import random
import numpy as np
from os.path import *
from os import listdir
from sklearn.feature_extraction.text import TfidfVectorizer
from urllib.request import urlopen


URL = "http://people.csail.mit.edu/jrennie/20Newsgroups/20news-bydate.tar.gz"
ARCHIVE_NAME = "20news-bydate.tar.gz"
TRAIN_FOLDER = "20news-bydate-train"
TEST_FOLDER = "20news-bydate-test"
PICKLED_STRING = "pickled_data"
PARTITION_STRING = "partition-"
CLUSTER_STRING = "clusters"
NUM_CLUSTERS = 5
DATA_DIR = "data"

def load_files(container_path, encoding=None):

    filenames = []

    folders = [f for f in sorted(listdir(container_path)) if isdir(join(container_path, f))]

    for label, folder in enumerate(folders):
        folder_path = join(container_path, folder)
        documents = [join(folder_path, d) for d in sorted(listdir(folder_path))]
        filenames.extend(documents)

    filenames = np.array(filenames)

    data = []
    for filename in filenames:
        with open(filename, 'rb') as f:
            data.append(f.read())
    if encoding is not None:
        data = [d.decode(encoding) for d in data]

    return data

def downloadFiles(target_dir):

    archive_path = os.path.join(target_dir, ARCHIVE_NAME)
    train_path = os.path.join(target_dir, TRAIN_FOLDER)
    test_path = os.path.join(target_dir, TEST_FOLDER)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if os.path.exists(archive_path):
        os.remove(archive_path)

    opener = urlopen(URL)
    with open(archive_path, 'wb') as f:
        f.write(opener.read())

    tarfile.open(archive_path, "r:gz").extractall(path=target_dir)
    os.remove(archive_path)

    total_data = dict(train=load_files(train_path, encoding='latin1'),
                test=load_files(test_path, encoding='latin1'))

    with open(target_dir + "/" + PICKLED_STRING + ".pkl", "wb") as f:
        pickle.dump(total_data, f)

    print("Files Downloaded.")

def loadFromPickle(tar_dir, testing=0, shuffle=True):

    with open(tar_dir + "/" + PICKLED_STRING + ".pkl", "rb") as f:
        total_data = pickle.load(f)

    if testing:
        data = total_data['test']
        print(str(len(data)) + " test instances loaded.")
    else:
        data = total_data['train']
        print(str(len(data)) + " train instances loaded.")

    if shuffle:
        np.random.shuffle(data)

    return data

def createVectors(tar_dir):

    data_train = loadFromPickle(tar_dir)
    # data_test = loadFromPickle(tar_dir, 1)

    vectorizer = TfidfVectorizer(min_df=1)
    X_train = vectorizer.fit_transform(data_train)
    # X_test = vectorizer.transform(data_test['data'])
    print("Train Matrix shape: " + str(X_train.shape))
    # print("Test Matrix shape: " + str(X_test.shape))

    totaltrain = X_train.shape[0]
    cluster_docs = []
    chunk = totaltrain // NUM_CLUSTERS
    print("Docs in each partition: " + str(chunk))
    start = 0
    end = chunk

    for i in range(NUM_CLUSTERS):
        if end >= totaltrain:
            end = -1
        cdata = X_train[start:end]
        with open(tar_dir + "/" + PARTITION_STRING + str(i) + ".pkl", "wb") as f:
            pickle.dump(cdata, f)
        cdoc = random.choice(range(start, end))
        cluster_docs.append(X_train[cdoc])
        start += chunk
        end += chunk

    with open(tar_dir + "/" + CLUSTER_STRING + ".pkl", "wb") as f:
        pickle.dump(cluster_docs, f)


if __name__ == '__main__':
    downloadFiles(DATA_DIR)
    createVectors(DATA_DIR)