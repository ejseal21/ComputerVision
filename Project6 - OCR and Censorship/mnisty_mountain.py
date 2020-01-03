#task 1
#Ethan Seal and Henry Doud
#CS365 Project 5
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from matplotlib import pyplot as plt
import numpy as np
import csv
import sys



def main(argv):
	batch_size = 128
	num_classes = 62
	epochs = 200
	print('USAGE: python3 mnisty_mountain.py <train.csv> <test.csv> (categories first column)')

	# input image dimensions
	img_rows, img_cols = 28, 28

	# get our file
	file = open(argv[1], 'rU')
	# read it with the csv module
	csv_reader = csv.reader(file)

	# loop over the lines
	for line in csv_reader:
		# temp_line allows self.data to be a 2d matrix
		temp_line = []
		for i in range(1, len(line)):
			temp_line.append(float(line[i]))
		x_train.append(temp_line)
		y_train.append(line[0])
	x_train = np.array(x_train)
		
	# get our file
	file = open(argv[2], 'rU')
	# read it with the csv module
	csv_reader = csv.reader(file)

	# loop over the lines
	for line in csv_reader:
		temp_line = []
		for i in range(1, len(line)):
			temp_line.append(float(line[i]))
		x_test.append(temp_line)
		y_test.append(line[0])
	x_test = np.array(x_test)

	# the data, split between train and test sets

	if K.image_data_format() == 'channels_first':
		x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
		x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
		input_shape = (1, img_rows, img_cols)
	else:
		x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
		x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
		input_shape = (img_rows, img_cols, 1)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
		          optimizer=keras.optimizers.Adam(),
		          metrics=['accuracy'])

	for i in range(epochs):
		print("epschwach:",i)
		model.fit(x_train, y_train,
			      batch_size=batch_size,
			      verbose=1,
			      validation_data=(x_test, y_test))
		score = model.evaluate(x_test, y_test, verbose=0)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])
		score = model.evaluate(x_train, y_train, verbose=0)
		print('Train loss:', score[0])
		print('Train accuracy:', score[1])
	model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

if __name__ == '__main__':
	main(sys.argv)
