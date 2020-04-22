import numpy as np
def kfold_split(data,targets,k):
    from random import randrange
    folds = np.empty([k, int(data.shape[0]/k), data.shape[1]])
    fold_targets = np.empty([k, int(targets.shape[0]/k)])
    fold_split = 1/k
    fold_size = int(fold_split * len(data))
    data_copy = list(data)
    targets_copy = list(targets)
    for i in range(k):
        this_fold = []
        this_target = []
        index = []
        while len(index) < fold_size:
            index.append(randrange(len(data_copy)))
            this_fold.append(data_copy.pop(index[-1]))
            this_target.append(targets_copy.pop(index[-1]))
        folds[i] = np.asarray(this_fold)
        this_target = np.asarray(this_target)
        fold_targets[i] =  np.reshape(this_target,(this_target.shape[0],))
    return folds, fold_targets