#task 2
#Ethan Seal and Henry Doud
#CS365 Project 5
import keras
import keras.preprocessing
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras import backend as K
from matplotlib import pyplot as plt
import numpy as np
import cv2

model = keras.models.load_model('my_model.h5')
layers = model.layers
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def first_one(): 
	
	weights = layers[0].get_weights()[0]
	print(weights.shape)
	f = plt.figure()
	for i in range(weights.shape[3]):
		display_filter = np.zeros((3,3))
		for j in range(weights.shape[0]):
			for k in range(weights.shape[1]):
				display_filter[j,k] = weights[j,k,0,i] 
		print(i//8, i%4)
		f.add_subplot(8,4,i+1)
		filtered = cv2.filter2D(x_train[0], -1, display_filter)
		# plt.imshow(display_filter)
		plt.imshow(filtered)

	plt.show(block=True)

def second_one():
	first_layer_model = keras.Model( inputs=model.input, outputs=model.get_layer(index=1).output )

	f = plt.figure()
	res = first_layer_model.predict(x_train[0].reshape(1, 28, 28, 1), batch_size=1)
	for i in range(res.shape[3]):
		f.add_subplot(8,4,i+1)
		plt.imshow(res[0,:,:,i])

	plt.show(block=True)

def third_one():
	#change which layer by changing the index argument
	layer_model = keras.Model( inputs=model.input, outputs=model.get_layer(index=3).output)
	f = plt.figure()
	#change x_train[index] to change which number it's doing it on
	res = layer_model.predict(x_train[2].reshape(1, 28, 28, 1), batch_size=1)
	print(y_train[2])
	for i in range(32):
		f.add_subplot(8,4,i+1)
		plt.imshow(res[0,:,:,i])

	plt.show(block=True)


def main():
	third_one()

if __name__ == "__main__":
	main()