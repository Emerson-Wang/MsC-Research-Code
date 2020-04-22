import numpy as np
import h5py
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, BatchNormalization
from keras.utils import to_categorical
from keras.models import load_model
from kfold import kfold_split
import matplotlib.pyplot as plt
from scipy import stats

#post-op = [0, 1, 1, 0, 0, 1, 1, 1, 0, 1]
# DATA INITIALIZATION
x = np.load('cardiac_records-TEST.npy')
x_image = np.load('testing_vals.npy')
y = np.load('demog_test_y.npy')
demog = np.load('demog_test.npy')
model1 = load_model('desi_cnn.h5')
model2 = load_model('desi_discrim.h5')
x = np.reshape(x,[x.shape[0],x.shape[1],1])
m1_output = model1.predict(x)
n = m1_output.shape[0]
x = np.zeros((1,1))
idx = 0
for val in np.unique(x_image):
    val_idx = np.where(x_image == val)
    cur = m1_output[val_idx]
    cur = np.sum(cur)/(cur.shape[0])
    x[idx] = cur
    idx += 1
new_x = np.append(demog[:,0:2], demog[:,3:4], axis=1)
new_x = np.append(new_x,x,axis=1)
#new_x = new_x[:,2:6]
#new_x = x
print(model2.metrics_names)
print(model2.evaluate(new_x,y,verbose=1,batch_size=50))
pred = model2.predict(new_x)
#pred[pred>0.5] = 1
#pred[pred<=0.5] = 0
np.savetxt('results_d.txt',pred,fmt='%f')
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
print('DONE')
