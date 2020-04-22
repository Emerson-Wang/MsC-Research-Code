import numpy as np
from keras.models import Sequential
from keras.layers import LocallyConnected1D, Dense, Conv1D, Flatten, BatchNormalization, Dropout
from keras.utils import to_categorical
from keras.models import load_model
import tensorflow as tf
import math
import h5py
from kfold import kfold_split
from keras.optimizers import adam
from keras.constraints import MaxNorm
import training_GPU
from scipy import stats

def main():
	x_desi = np.load('cardiac_records.npy')
	x_image = np.load('training_vals.npy')
	### pre-op afib
	#targets = [1, 0, 1, 0, 0 ,0, 0, 1, 0, 1]
	### myocytolysis
	#targets = [0, 0, 1, 0, 1, 0, 0, 0, 1, 0]
	### nuclear hypertrophy
	#targets = [1, 1, 0, 1, 0, 1, 1, 1, 0, 0]
	### post-op afib
	targets = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 1])
	x_desi = np.reshape(x_desi,[x_desi.shape[0],x_desi.shape[1],1])
	split = 10
	dem = np.load('demog_train.npy')
	demog = np.empty((0,5))
	samples = 10
	idx = 0
	x = np.zeros((np.unique(x_image).size*samples,1))
	y = np.zeros((np.unique(x_image).size*samples,1))
	#training_GPU.main()
	for i in range(samples):
		model1 = load_model('desi_cnn.h5')
		m1_output = model1.predict(x_desi)
		#n = m1_output.shape[0]
		demog = np.append(demog,dem,axis=0)
		for val in np.unique(x_image):
			val_idx = np.where(x_image == val)
			cur = m1_output[val_idx]
			cur[cur>0.5] = 1
			cur[cur<=0.5] = 0
			cur = np.sum(cur)/(cur.shape[0])
			x[idx] = cur
			y[idx] = targets[val]
			idx += 1
	new_x = np.append(demog[:,0:2], demog[:,3:4], axis=1)
	new_x = np.append(new_x, x,axis=1)
	#new_x = new_x[:,3:6]
	#new_x = x
	model2 = Sequential()
	model2.add(Dense(units=max(int(new_x.shape[1]/2),1), input_shape=(new_x.shape[1],), activation ='relu'))
	model2.add(Dense(units=1, input_shape=(new_x.shape[1],), activation='sigmoid'))
	adm = adam(lr=0.001, beta_1=0.9, beta_2=0.98)
	model2.compile(optimizer=adm,loss='mean_squared_error',metrics=['accuracy'])
	print('TRAINING')
	epochs = 1000
	for ep in range(epochs):
		x_split,y_split = kfold_split(new_x,y,split)
		print("epoch: " + str(ep))
		for k in range(split):
			train = np.ones((split,),bool)
			train[k] = False
			x_train = x_split[train,:,:]
			y_train = y_split[train,:]
			x_train = np.reshape(x_train,[x_train.shape[0]*x_train.shape[1],x_train.shape[2]])  
			y_train = np.reshape(y_train,[y_train.shape[0]*y_train.shape[1]])
			x_val = x_split[k,:,:]
			y_val = y_split[k,:]
			x_val = np.reshape(x_val,[x_val.shape[0],x_val.shape[1]])  
			model2.fit(x_train,y_train,validation_data=(x_val,y_val), epochs=1, shuffle=True, verbose=2)
	model2.save('desi_discrim.h5')
	print("Done")
#main()