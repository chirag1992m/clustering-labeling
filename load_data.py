import os
import tarfile
import pickle
import random
import numpy as np
from os.path import *
from os import listdir
from sklearn.feature_extraction.text import TfidfVectorizer
from urllib.request import urlopen
from . import constants


URL = "http://people.csail.mit.edu/jrennie/20Newsgroups/20news-bydate.tar.gz"
ARCHIVE_NAME = "20news-bydate.tar.gz"
TRAIN_FOLDER = "20news-bydate-train"
TEST_FOLDER = "20news-bydate-test"
PICKLED_STRING = "pickled_data"
PARTITION_STRING = ""
DATA_DIR = "clustering-labeling/data/20_newsgroup_raw"
TARGET_DIR = "clustering-labeling/20_newsgroup"

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

def loadFromPickle(tar_dir, shuffle=True):

    with open(tar_dir + "/" + PICKLED_STRING + ".pkl", "rb") as f:
        total_data = pickle.load(f)

    data = total_data['test']
    data.extend(total_data['train'])
    print(str(len(data)) + " instances loaded.")

    if shuffle:
        np.random.shuffle(data)

    return data

def createVectors(input_dir, target_dir):

    data_train = loadFromPickle(input_dir)
    
    vectorizer = TfidfVectorizer(min_df=1)
    X_train = vectorizer.fit_transform(data_train)
    print("Train Matrix shape: " + str(X_train.shape))
    
    totaltrain = X_train.shape[0]

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    with open(target_dir + "/0.out", "wb") as f:
        for idx in range(constants.NUM_CLUSTERS):
            choice = int(random.random() * totaltrain)
            pickle.dump((idx, X_train[choice]), f)

    chunk = totaltrain // constants.NUM_PARTITIONS
    print("Docs in each partition: " + str(chunk))
    start = 0
    end = chunk

    for i in range(constants.NUM_PARTITIONS):
        if end >= totaltrain:
            end = -1
        cdata = X_train[start:end]
        with open(target_dir + "/" + PARTITION_STRING + str(i) + ".in", "wb") as f:
            pickle.dump(cdata, f)
        start += chunk
        end += chunk


if __name__ == '__main__':
    downloadFiles(DATA_DIR)
    createVectors(DATA_DIR, TARGET_DIR)