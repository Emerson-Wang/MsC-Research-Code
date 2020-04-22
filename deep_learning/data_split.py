import numpy as np
from kfold import kfold_split

# DATA INITIALIZATION
file_name = 'cardiac_processed'
file_ext = '.txt'
labels = np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 1])
#labels = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 1])
training = []
testing = [0, 1, 2, 3, 4, 5, 6, 7, 9]
folds = 1

x = []
y = []
for i in range(10):
    ys = []
    print(file_name + str(i) + file_ext)
    data = np.genfromtxt(file_name + str(i) + file_ext, delimiter=',')
    spectra_sums = np.sum(data,axis=1)
    max_tic = max(spectra_sums)
    for j in range(spectra_sums.shape[0]):
        ys.append(labels[i])
    cur, tg = kfold_split(np.asarray(data), np.asarray(ys), folds)
    x.append(cur)
    y.append(tg)

x_train = []
y_train = []
x_test = []
y_test = []
for i in range(10):
    if i in training:
        for n in range(folds):
            for j in range(x[i][n].shape[0]):
                x_train.append(x[i][n][j])
                y_train.append(y[i][n][j])
    if i in testing:
        for n in range(folds-1):
            for j in range(x[i][n].shape[0]):
                x_test.append(x[i][n][j])
                y_test.append(y[i][n][j])
        for j in range(x[i][folds-1].shape[0]):
            x_train.append(x[i][folds-1][j])
            y_train.append(y[i][folds-1][j])
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
np.save('cardiac_records',x_train)
np.save('cardiac_targets',y_train)
np.save('cardiac_records-TEST',x_test)
np.save('cardiac_targets-TEST',y_test)