#task 1
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


img_rows, img_cols = 28, 28
model = keras.models.load_model('my_model.h5')
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

res = model.predict(x_test[:10], batch_size=10)
print("res:", res)
target = y_test[:10]
print("\n\n\ntarget:",target)

for i in range(len(res)):
	print("predicted:", res[i].tolist().index(max(res[i].tolist())))
	print("target:", target[i])

numbers = []
for i in range(10):
	img_path = str(i)+".jpg"
	print(img_path)
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, img = cv2.threshold( img, 174, 255, cv2.THRESH_BINARY)
	img = cv2.resize(img, (28,28))
	img = np.invert(img)	
	numbers.append(img)



numbers = np.array(numbers)
numbers = numbers.reshape((numbers.shape[0], img_rows, img_cols, 1))
	
res = model.predict(numbers, batch_size=10)
for i in range(len(res)):
	print("predicted:", res[i].tolist().index(max(res[i].tolist())))
	print("target:", i)
