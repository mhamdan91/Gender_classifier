import os
import numpy as np
import random
from sklearn.model_selection import train_test_split

'''
This file shuffles the datasets. 
Validation dataset is splitted from the training dataset.
Validation set has the same size as test set.

'''


sep = os.sep
HOME = os.getcwd()
MAIN = HOME+sep+r'combined'

test_labels = np.load('testing_labels.npy')
test_data = np.load('testing_data.npy')
test_names = np.load('testing_names_images.npy')
train_labels = np.load('training_labels.npy')
train_data = np.load('training_data.npy')
train_names = np.load('training_names_images.npy')


t = list(zip(test_labels, test_data, test_names))
random.shuffle(t)
test_labels, test_data, test_names = zip(*t)

tr = list(zip(train_labels, train_data, train_names))
random.shuffle(tr)
tr, v = train_test_split(tr, test_size=0.125)  # same as test data

train_labels, train_data, train_names = zip(*tr)
validate_labels, validate_data, validate_names = zip(*v)
np.save('testing_data_shuffled.npy', test_data)
np.save('testing_labels_shuffled.npy', test_labels)
np.save('testing_names_images_shuffled.npy', test_names)

np.save('training_data_shuffled.npy', train_data)
np.save('training_labels_shuffled.npy', train_labels)
np.save('training_names_images_shuffled.npy', train_names)

np.save('validation_data_shuffled.npy', validate_data)
np.save('validation_labels_shuffled.npy', validate_labels)
np.save('validation_names_images_shuffled.npy', validate_names)