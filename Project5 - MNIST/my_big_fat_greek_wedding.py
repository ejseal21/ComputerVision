#task 3
#Ethan Seal and Henry Doud
#CS365 Project 5
import os
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
import csv
from scipy import stats
import sys

model = keras.models.load_model('my_model.h5')
layers = model.layers
(x_train, y_train), (x_test, y_test) = mnist.load_data()
categories = []

def read_stuff(filename, directory):
	files = sorted(os.listdir(directory))
	filename = open(filename, 'w')
	writer = csv.writer(filename)
	headers = ["val_"+str(i) for i in range(784)]
	writer.writerow(headers)
	for i in range(len(files)):		
		img = cv2.imread(directory + "/" + files[i])
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (28,28))
		img = img.flatten().tolist()
		writer.writerow(img)

def write_cats(filename, directory):
	files = sorted(os.listdir(directory))
	cats = directory.split('_')
	categories.extend(cats)
	f = open(filename, 'w')
	writer = csv.writer(f)
	writer.writerow(['categories'])
	for i in range(len(files)):
		# print(files[i])
		writer.writerow([cats.index(files[i].split('_')[0])])

def predict_stuff(data, cats):
	data = open(data, 'r')
	cats = open(cats, 'r')
	reader = csv.reader(data)
	cat_reader = csv.reader(cats)
	sixth_layer_model = keras.Model( inputs=model.input, outputs=model.get_layer(index=6).output )

	imgs =  [r for r in reader]
	imgs.pop(0) # remove header
	res = sixth_layer_model.predict(np.array(imgs).reshape(len(imgs),28,28,1), batch_size=len(imgs)) # 27,128
	catsr = [r for r in cat_reader]
	catsr.pop(0) # remove header
	cats = [int(r[0]) for r in catsr]
	
	exs = []
	for i in range(max(cats) + 1):
		exs.append(res[cats.index(i)])
	ssd = []

	for i in range(res.shape[0]):
		ssd.append(np.sum(((exs-res[i,:])**2),1))
	ssd = np.array(ssd).T

	# print(ssd)
	# print(cats)
	return res, cats

def knn_classifier(img, k, res, cats):
	sixth_layer_model = keras.Model( inputs=model.input, outputs=model.get_layer(index=6).output )
	res_img = sixth_layer_model.predict(np.array(img).reshape(1,28,28,1), batch_size=1)
	ssd = []
	for i in range(res.shape[0]):
		ssd.append(np.sum(((res_img-res[i,:])**2),1))
	ssd = np.array(ssd).T
	classes = []
	for i in range(k):
		index = np.argmin(ssd,1)[0]		
		classes.append(cats[index])
		
		ssd[0,index] = float("inf")
	cat = stats.mode(classes)[0][0]
	return categories[cat]

def main(argv):
	print("\n\n\n")
	#if the user supplies only a filepath
	if len(argv) == 2:
		read_stuff("test.csv", "alpha_beta_gamma_delta_epsilon_eta")
		write_cats("test_cats.csv", "alpha_beta_gamma_delta_epsilon_eta")	
		res, cats = predict_stuff("test.csv","test_cats.csv")
		img = cv2.imread(argv[1])
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (28,28))	
		print("It is predicted to be", knn_classifier(img, 8, res, cats))
		
		#if the user supplies a filepath and a directory
	elif len(argv) == 3:
		read_stuff("test.csv", argv[2])
		write_cats("test_cats.csv", argv[2])
		res, cats = predict_stuff("test.csv","test_cats.csv")
		img = cv2.imread(argv[1])
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (28,28))	
		print("It is predicted to be", knn_classifier(img, 8, res, cats))

	#if the user just wants to see something happen
	else:
		read_stuff("test.csv", "alpha_beta_gamma_delta_epsilon_eta")
		write_cats("test_cats.csv", "alpha_beta_gamma_delta_epsilon_eta")
		res, cats = predict_stuff("test.csv","test_cats.csv")
		

		files = sorted(os.listdir("alpha_beta_gamma_delta_epsilon_eta"))
		for file in files:
			img = cv2.imread( "alpha_beta_gamma_delta_epsilon_eta" + "/" + file )
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img = cv2.resize(img, (28,28))	
			print(file , "is predicted to be", knn_classifier(img, 4, res, cats))


if __name__ == "__main__":
	main(sys.argv)
