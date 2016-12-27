import os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import numpy as np

def loadData(path):
	pathx = path + "/datax"
	pathy = path + "/datay"
	files = os.listdir(pathx)
	while 1:
		for file in files:
			if not os.path.isdir(file):
				datax = np.load(pathx + "/" + file)
				datay = np.load(pathy + "/" + file)
				yield datax['dx'], datay['dy']

path = "sampleData"

seed = 7
np.random.seed(seed)

model = Sequential()
model.add(Convolution2D(64, 5, 5, border_mode='same', input_shape=(7, 19, 19), dim_ordering='th'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
model.add(Flatten())
model.add(Dense(361, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model.fit_generator(loadData(path), samples_per_epoch=4096, nb_epoch=5)

model.save_weights('model/weights.hd5', overwrite=True)

with open('model/model.yml', 'w') as yml:
    model_yaml = model.to_yaml()
    yml.write(model_yaml)
