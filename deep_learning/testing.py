import numpy as np
import h5py
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, BatchNormalization
from keras.utils import to_categorical
from keras.models import load_model
from kfold import kfold_split
import matplotlib.pyplot as plt
import copy

# DATA INITIALIZATION
x = np.load('cardiac_records-TEST.npy')
y = np.load('cardiac_targets-TEST.npy')
model = load_model('desi_cnn.h5')
x = np.reshape(x,[x.shape[0],x.shape[1],1])
print(model.metrics_names)
print(model.evaluate(x,y,verbose=1,batch_size=50))
pred = model.predict(x)
pred_thresh = copy.deepcopy(pred)
pred_thresh[pred>0.5] = 1
pred_thresh[pred<=0.5] = 0
np.savetxt('results.txt',pred,fmt='%f')
np.savetxt('y_test.txt',y)
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
#sens = sens/classes[1]
#spec = spec/classes[0]
print("Sensitivity: " + str(sens))
print("Specificity: " + str(spec))
print("Predicted: " + str(np.sum(pred)/(pred.shape[0])))
print("Predicted thresholded: " + str(np.sum(pred_thresh)/(pred.shape[0])))
print('DONE')
