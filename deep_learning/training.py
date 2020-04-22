import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, BatchNormalization, Dropout
from keras.utils import to_categorical
from keras.models import load_model
import tensorflow as tf
import math
import h5py
from kfold import kfold_split

# DATA INITIALIZATION
x = np.load('cardiac_records.npy')
y = np.load('cardiac_targets.npy')
x,y = kfold_split(x,y,1)
x = x[0,:,:]
y = y[0,:]
x = np.reshape(x,[x.shape[0],x.shape[1],1])  
#x = x[0:10000,:,:]
#y = y[0:10000]
#print(y)
print(x.shape)
print(y.shape)
print(np.sum(y))
# CNN INITIALIZATION
model = Sequential()
model.add(BatchNormalization())
# Convolution 1:
model.add(Conv1D(filters=64, kernel_size=5, input_shape=(x.shape[1],x.shape[2]), activation='relu', padding='same'))
model.add(Dropout(rate=0.5))
# Fully connected dense layer
model.add(Dense(units=x.shape[1], activation='relu'))
# Flatten
model.add(Flatten())
# Add a single output logit layer:
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
print('TRAINING')
model.fit(x, y, validation_split=0.25, epochs=50, batch_size=150, shuffle=True, verbose=2)
model.save('desi_cnn.h5')