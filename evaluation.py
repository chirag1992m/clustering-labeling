import pickle

def matchk(truth_file, pred_files):
    with open(truth_file) as f:
        true_labels = pickle.load(f)

    count = 0
    pred_acc = []
    for ind, pfile in enumerate(pred_files):
        with open(pfile) as f:
            pred_labels = pickle.load(f)

        for label in pred_labels:
            if label.find(true_labels[ind]):
                count += 1
        pred_acc.append(count)

    print(pred_acc)