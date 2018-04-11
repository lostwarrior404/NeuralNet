import cPickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from hep_ml.nnet import MLPMultiClassifier
# code from cifar code unpickle example

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo)
	return dict


db1=unpickle('./data/data_batch_1')
db2=unpickle('./data/data_batch_2')
db3=unpickle('./data/data_batch_3')
db4=unpickle('./data/data_batch_4')
db5=unpickle('./data/data_batch_5')
test=unpickle('./data/test_batch')
train_data=[]
train_labels=[]
for i in range(1,6):
	db=unpickle('./data/data_batch_'+str(i))
	for data,label in zip(db['data'],db['labels']):
		train_data.append(data)
		train_labels.append(label)
train_data=np.array(train_data)
train_labels=np.array(train_labels)

test_data=np.array(test['data'])
test_labels=np.array(test['labels'])

# print(train_data.shape)
# print(train_labels.shape)

train_data=train_data.reshape((50000,1024,3))
train_data=np.average(train_data, axis=-1)
test_data=test_data.reshape((test_data.shape[0],1024,3))
test_data=np.average(test_data, axis=-1)

# train_data=np.ravel(train_data)
# test_data=np.ravel(test_data)

print(train_data.shape)
print(test_data.shape)

print()
# clf = MLPClassifier(hidden_layer_sizes=(50), max_iter=2, learning_rate_init=0.1, activation='identity', verbose=True)
# clf.fit(train_data, train_labels)
# predictions=clf.predict(test_data)
# print()
# print(accuracy_score(test_labels, predictions))

# clf=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
# clf.fit(train_data, train_labels)
# print(clf.score(test_data, test_labels))


# clf=BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
# clf.fit(train_data, train_labels)
# print(clf.score(test_data, test_labels))



base_network = MLPMultiClassifier(layers=[50])
# clf=AdaBoostClassifier()

clf = AdaBoostClassifier(base_estimator=base_network, n_estimators=5)
clf.fit(train_data, train_labels)
print("N-E=5")
print(clf.score(test_data, test_labels))

clf = AdaBoostClassifier(base_estimator=base_network, n_estimators=15)
clf.fit(train_data, train_labels)
print("N-E=15")
print(clf.score(test_data, test_labels))


clf = BaggingClassifier(base_estimator=base_network, n_estimators=5)
clf.fit(train_data, train_labels)
print("N-E=15")
print(clf.score(test_data, test_labels))

clf = BaggingClassifier(base_estimator=base_network, n_estimators=15)
clf.fit(train_data, train_labels)
print("N-E=15")
print(clf.score(test_data, test_labels))

# 

print("Change Base Neural Networl")
base_network = MLPMultiClassifier(layers=[400,150])

clf = AdaBoostClassifier(base_estimator=base_network, n_estimators=5)
clf.fit(train_data, train_labels)
print("N-E=5")
print(clf.score(test_data, test_labels))

clf = BaggingClassifier(base_estimator=base_network, n_estimators=5)
clf.fit(train_data, train_labels)
print("N-E=15")
print(clf.score(test_data, test_labels))


