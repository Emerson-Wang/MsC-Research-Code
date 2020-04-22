import numpy as np
from scipy import stats
from sklearn import preprocessing
import random

# DATA INITIALIZATION
file_name = 'cardiac_processed'
file_ext = '.txt'
### pre-op afib
labels = np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 1])
### post-op afib
labels = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 1])
afib_labels = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 1])
### myocytolysis
#labels = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 1])
### nuclear hypertrophy
#labels = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 0])
### interstitial fibrosis
#labels = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 1])

training = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
testing = [0]

minority = 1
if(sum(labels)>labels.shape[0]/2):
    minority = 0
x = []
y = []
demog_train = []
demog_train_y = []
demog_test = []
demog_test_y = []
training_vals = []
testing_vals = []
samples = []
sample_sizes = []
max_size = 0
demog = np.genfromtxt('demographic.txt',delimiter=',')
data = np.genfromtxt(file_name + str(0) + file_ext, delimiter=',')
samples_f = np.empty((0,data.shape[1]))
for i in range(10):
        data = np.genfromtxt(file_name + str(i) + file_ext, delimiter=',')
        data_size = data.shape[0]
        sample_sizes.append(data_size)
        if data_size > max_size:
            max_size = data_size
        samples.append(data)
        samples_f = np.append(samples_f,data,axis=0)
#samples_f = preprocessing.normalize(samples_f,norm='max',axis=0)
#start = 0
#for i in range(10):
    #samples[i] = samples_f[start:start+sample_sizes[i],:]
    #start = start+sample_sizes[i]
for i in training:
        print(file_name + str(i) + file_ext)
        data = samples[i]
        spectra_sums = np.sum(data,axis=1)
        max_tic = max(spectra_sums)
        this_size = data.shape[0]
        idx = 0
        demog_train.append(demog[i])
        demog_train_y.append(afib_labels[i])
        for j in range(max_size):
            if(idx == this_size):
                idx = 0
            x.append(data[idx]/np.max(data[idx]))
            #x.append(data[idx])
            y.append(labels[i])
            training_vals.append(i)
            idx += 1
x = np.asarray(x) 
y = np.asarray(y)
minority_idx = np.where(y == int(minority))
minority_set = np.append(x[minority_idx], x[minority_idx], axis=0)
minority_set = np.append(minority_set, x[minority_idx], axis=0)
np.random.shuffle(minority_set)
diff = abs(np.sum(y,axis=0) - y.shape[0]/2) * 2
#x = np.append(x, minority_set[0:int(diff),:], axis=0)
#y = np.append(y, np.full((int(diff),), int(minority)), axis=0)
x_test = []
y_test = []
for i in testing:
        print(file_name + str(i) + file_ext)
        data = np.genfromtxt(file_name + str(i) + file_ext, delimiter=',')
        spectra_sums = np.sum(data,axis=1)
        max_tic = max(spectra_sums)
        demog_test.append(demog[i])
        demog_test_y.append(afib_labels[i])
        for j in range(spectra_sums.shape[0]):
            x_test.append(data[j]/np.max(data[j]))
            #x_test.append(data[j])
            y_test.append(labels[i])
            testing_vals.append(i)
            
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
testing_vals = np.asarray(testing_vals)
training_vals = np.asarray(training_vals)
np.save('cardiac_records',x)
np.save('cardiac_targets',y)
np.save('demog_train',demog_train)
np.save('demog_test',demog_test)
np.save('demog_test_y',demog_test_y)
np.save('cardiac_records-TEST',x_test)
np.save('cardiac_targets-TEST',y_test)
np.save('training_vals',training_vals)
np.save('testing_vals',testing_vals)