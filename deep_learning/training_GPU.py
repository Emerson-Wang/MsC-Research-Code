import numpy as np
from keras.models import Sequential
from keras.layers import LocallyConnected1D, Dense, Conv1D, Flatten, BatchNormalization, Dropout
from keras.utils import to_categorical
from keras.models import load_model
from keras.constraints import MaxNorm
import tensorflow as tf
import math
import h5py
from kfold import kfold_split
from keras.optimizers import adam

def main():
	# DATA INITIALIZATION
	split = 10
	x = np.load('cardiac_records.npy')
	y = np.load('cardiac_targets.npy')
	x_split,y_split = kfold_split(x,y,split)
	print(x.shape)
	print(sum(y))
	# CNN INITIALIZATION
	model = Sequential()
	model.add(BatchNormalization())
	model.add(Dropout(rate=0))
	# Convolution 1:
	model.add(Conv1D(filters=9, kernel_size=5, input_shape=(x_split.shape[2],1), kernel_constraint=MaxNorm(3), kernel_initializer='uniform', activation='relu', padding='valid'))
	model.add(Dropout(rate=0))
	# Convolution 2:
	model.add(Conv1D(filters=3, kernel_size=11, input_shape=(x_split.shape[2],1), kernel_constraint=MaxNorm(3), kernel_initializer='uniform', activation='relu', padding='valid'))
	model.add(Dropout(rate=0))
	model.add(Flatten())

	# Fully connected dense layer
	model.add(Dense(units=x_split.shape[2], kernel_initializer='uniform',  kernel_constraint=MaxNorm(3), activation='relu'))
	model.add(Dropout(rate=0))

	# Add a single output logit layer:
	model.add(Dense(units=1, kernel_initializer='uniform',  kernel_constraint=MaxNorm(3), activation='sigmoid'))
	adm = adam(lr=0.01, beta_1=0.8, beta_2=0.98,)
	model.compile(optimizer=adm, loss='mean_squared_error', metrics=['accuracy'])
	print('TRAINING')
	epochs = 3
	for ep in range(epochs):
		print("epoch: " + str(ep))
		for k in range(split):
			x_split,y_split = kfold_split(x,y,split)
			train = np.ones((split,),bool)
			train[k] = False
			x_train = x_split[train,:,:]
			y_train = y_split[train,:]
			x_train = np.reshape(x_train,[x_train.shape[0]*x_train.shape[1],x_train.shape[2],1])
			y_train = np.reshape(y_train,[y_train.shape[0]*y_train.shape[1]])
			x_val = x_split[k,:,:]
			y_val = y_split[k,:]
			x_val = np.reshape(x_val,[x_val.shape[0],x_val.shape[1],1])
			model.fit(x_train,y_train,validation_data=(x_val,y_val), epochs=1, shuffle=True, verbose=2)
	model.save('desi_cnn.h5')
	print("Done")

main()