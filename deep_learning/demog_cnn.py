import numpy as np
from scipy import stats
from sklearn import preprocessing
import random
import h5py
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, BatchNormalization
from keras.utils import to_categorical
from keras.models import load_model
from kfold import kfold_split
import training_GPU
import train_discriminator
import matplotlib.pyplot as plt
import copy

outfile = 'afib_demog_test.txt'
trials = 1

def import_data(testing_val):
    # DATA INITIALIZATION
    file_name = 'cardiac_processed'
    file_ext = '.txt'
    ### nuclear hypertrophy
    labels = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 0])
    ### poaf
    afib_labels = [0, 1, 1, 0, 0, 1, 1, 1, 0, 1]
    samp = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    training = samp[samp!=testing_val]
    testing = [testing_val]
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
    for i in training:
            print(file_name + str(i) + file_ext)
            data = samples[i]
            spectra_sums = np.sum(data,axis=1)
            this_size = data.shape[0]
            idx = 0
            demog_train.append(demog[i])
            demog_train_y.append(afib_labels[i])
            for j in range(max_size):
                if(idx == this_size):
                    idx = 0
                x.append(data[idx]/np.max(data[idx]))
                y.append(labels[i])
                training_vals.append(i)
                idx += 1
    x = np.asarray(x) 
    y = np.asarray(y)
    x_test = []
    y_test = []
    for i in testing:
            print(file_name + str(i) + file_ext)
            data = np.genfromtxt(file_name + str(i) + file_ext, delimiter=',')
            spectra_sums = np.sum(data,axis=1)
            for j in range(spectra_sums.shape[0]):
                x_test.append(data[j]/np.max(data[j]))
                y_test.append(labels[i])
                testing_vals.append(i)
                demog_test.append(demog[i])
                demog_test_y.append(afib_labels[i])
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
def testing():
    # DATA INITIALIZATION
    x = np.load('cardiac_records-TEST.npy')
    y = np.load('demog_test_y.npy')
    demog = np.load('demog_test.npy')
    model0 = load_model('desi_cnn.h5')
    model = load_model('desi_discrim.h5')
    x = np.reshape(x,[x.shape[0],x.shape[1],1])
    pred0 = model0.predict(x)
    new_x = np.append(demog[:,0:2], demog[:,3:4], axis=1)
    new_x = np.append(new_x,pred0,axis=1)
    print(model.metrics_names)
    print(model.evaluate(new_x,y,verbose=1,batch_size=50))
    pred = model.predict(new_x)
    pred_thresh = copy.deepcopy(pred)
    pred_thresh[pred>0.5] = 1
    pred_thresh[pred<=0.5] = 0
    np.savetxt('demog_results.txt',pred,fmt='%f')
    np.savetxt('demog_y_test.txt',y)
    sens = 0
    spec = 0
    classes = [0,0]
    for i in range(y.shape[0]):
        target = int(y[i])
        predict = pred[i] > 0.5
        classes[target] += 1
        if predict == target:
            if target == 0:
                spec += 1
            elif target == 1:
                sens += 1
    
    prediction = np.sum(pred_thresh)/(pred.shape[0])
    print("Sensitivity: " + str(sens))
    print("Specificity: " + str(spec))
    print("Predicted " + str(prediction))
    print('DONE')
    return prediction

trial_results = np.empty((trials,10))
for i in range(trials):
    results = []
    for test in range(10):
        import_data(test)
        training_GPU.main()
        train_discriminator.main()
        testing_out = testing()
        results.append(testing_out)
    trial_results[i,:] = results
np.savetxt(outfile, trial_results)
