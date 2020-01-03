#task 4
#Ethan Seal and Henry Doud
#CS365 Project 5
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from matplotlib import pyplot as plt
import numpy as np

np.random.seed(42)

# #This stuff is to show the images 
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# f = plt.figure()
# f.add_subplot(1,2,1)
# plt.imshow(x_train[0])
# f.add_subplot(1,2,2)
# plt.imshow(x_train[1])
# plt.show(block=True)


batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

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

#changing the kernel size of the first convolution layer
for k in range(1, 10):
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(k, k), activation='relu', input_shape=input_shape))
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

	model.fit(x_train, y_train,
	          batch_size=batch_size, epochs = 10,
	          verbose=1,
	          validation_data=(x_test, y_test))
	score = model.evaluate(x_test, y_test, verbose=0)
	print('filter size of first convolution layer:', k, 'x', k)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	score = model.evaluate(x_train, y_train, verbose=0)
	print('Train loss:', score[0])
	print('Train accuracy:', score[1])

	# model.save('my_model' + str(k) + '.h5')  # creates a HDF5 file 'my_model.h5'

#changing the dropout rate of the first dropout layer
for k in range(10):
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.05 + k/10.0))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	# for j in range(10):
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adam(),
	              metrics=['accuracy'])

	model.fit(x_train, y_train,
	          batch_size=batch_size, epochs = 10,
	          verbose=1,
	          validation_data=(x_test, y_test))		
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Dropout rate of first layer:', k)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	score = model.evaluate(x_train, y_train, verbose=0)
	print('Train loss:', score[0])
	print('Train accuracy:', score[1])

#changing the dropout rate of the second dropout layer
for k in range(10):
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	# for j in range(10):
	model.add(Dropout(0.05 + k/10.0))
	model.add(Dense(10, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adam(),
	              metrics=['accuracy'])

	model.fit(x_train, y_train,
	          batch_size=batch_size, epochs = 10,
	          verbose=1,
	          validation_data=(x_test, y_test))
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Dropout rate of second layer:', k)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	score = model.evaluate(x_train, y_train, verbose=0)
	print('Train loss:', score[0])
	print('Train accuracy:', score[1])

	# model.save('dropout_model_1_' + str(k) + '.h5')  # creates a HDF5 file 'my_model.h5'

#changing the max pooling size
for k in range(1, 10):
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(k, k)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	# for j in range(10):
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adam(),
	              metrics=['accuracy'])

	model.fit(x_train, y_train,
	          batch_size=batch_size, epochs = 10,
	          verbose=1,
	          validation_data=(x_test, y_test))
	score = model.evaluate(x_test, y_test, verbose=0)
	print('MaxPooling2D size:', k)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	score = model.evaluate(x_train, y_train, verbose=0)
	print('Train loss:', score[0])
	print('Train accuracy:', score[1])

for k in range(1, 10):
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(2**k, activation='relu'))
	# for j in range(10):
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adam(),
	              metrics=['accuracy'])

	model.fit(x_train, y_train,
	          batch_size=batch_size, epochs = 10,
	          verbose=1,
	          validation_data=(x_test, y_test))
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Number of nodes in the dense layer:', 2**k)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	score = model.evaluate(x_train, y_train, verbose=0)
	print('Train loss:', score[0])
	print('Train accuracy:', score[1])

