import os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import model_from_yaml
import yaml
import numpy as np

seed = 7
np.random.seed(seed)

x=np.zeros((1,7,19,19))

with open('model.yml', 'r') as f:
	yml = yaml.load(f)
	model = model_from_yaml(yaml.dump(yml))
	model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
	model.load_weights('weights.hd5')

y = model.predict(x, batch_size=1, verbose=0)

print (y)

