import os
from scipy.misc import imread
import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
import operator
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from sklearn.ensemble import AdaBoostClassifier
from skimage import color
from hep_ml.nnet import MLPMultiClassifier
import pandas

def unpickle(file):
    with open(file, 'rb') as fo:
        dict1 = pickle.load(fo)
    return dict1

images2 = []
images = []
original = []
labels = []
ii = []
face_folder=os.listdir("./data")
'''for i in face_folder:
	images2.append(unpickle(i))'''
images2.append(unpickle('./data/data_batch_1'))
images2.append(unpickle('./data/data_batch_2'))
images2.append(unpickle('./data/data_batch_3'))
images2.append(unpickle('./data/data_batch_4'))
images2.append(unpickle('./data/data_batch_5'))
images2.append(unpickle('./data/test_batch'))

for i in images2:
	for j in i[b'data']:
		ii.append(np.average(j.reshape(1024,3),axis = 1))
	for j in i[b'labels']:
		labels.append(j)
train1 = []
test1 = []
train1_l = []
test1_l = []

for i in range(len(ii)):
	if(i < 0.7*len(ii)):
		train1.append(ii[i])
		train1_l.append(labels[i])
	else:
		test1.append(ii[i])
		test1_l.append(labels[i])

train1 = np.array(train1)
test1 = np.array(test1)
print(train1.shape)

base_network = MLPMultiClassifier(layers=[50])
classifier = AdaBoostClassifier(base_estimator=base_network, n_estimators=20)
classifier.fit(train1, train1_l)
y_pred = classifier.predict(images1)
print(accuracy_score(labels1, y_pred))

classifier = BaggingClassifier(base_estimator = base_network, n_estimators = 20)
classifier.fit(train1, train1_l)
y_pred = classifier.predict(images1)
print(accuracy_score(labels1, y_pred))

classifier = AdaBoostClassifier(base_estimator=base_network, n_estimators=5)
classifier.fit(train1, train1_l)
y_pred = classifier.predict(images1)
print(accuracy_score(labels1, y_pred))

classifier = BaggingClassifier(base_estimator = base_network, n_estimators = 15)
classifier.fit(train1, train1_l)
y_pred = classifier.predict(images1)
print(accuracy_score(labels1, y_pred))

base_network = MLPMultiClassifier(layers=[400,150])
classifier = AdaBoostClassifier(base_estimator=base_network, n_estimators=20)
classifier.fit(train1, train1_l)
y_pred = classifier.predict(images1)
print(accuracy_score(labels1, y_pred))

classifier = BaggingClassifier(base_estimator = base_network, n_estimators = 20)
classifier.fit(train1, train1_l)
y_pred = classifier.predict(images1)
print(accuracy_score(labels1, y_pred))

classifier = AdaBoostClassifier(base_estimator=base_network, n_estimators=5)
classifier.fit(train1, train1_l)
y_pred = classifier.predict(images1)
print(accuracy_score(labels1, y_pred))

classifier = BaggingClassifier(base_estimator = base_network, n_estimators = 15)
classifier.fit(train1, train1_l)
y_pred = classifier.predict(images1)
print(accuracy_score(labels1, y_pred))
