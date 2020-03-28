from __future__ import absolute_import, division, print_function
import tensorflow as tf
import os
import numpy as np

'''

'''
###############################################
tfe = tf.contrib.eager
sep = os.sep


# Load Data using pre-processed numpy data - Did it this way because the dataset is relatively small...
train_images = np.load(r'.'+sep+'data_files'+sep+'training_data_shuffled.npy')
train_labels = np.load(r'.'+sep+'data_files'+sep+'training_labels_shuffled.npy')
test_images = np.load(r'.'+sep+'data_files'+sep+'testing_data_shuffled.npy')
test_labels =np.load(r'.'+sep+'data_files'+sep+'testing_labels_shuffled.npy')
validate_images = np.load(r'.'+sep+'data_files'+sep+'validation_data_shuffled.npy')
validate_labels =np.load(r'.'+sep+'data_files'+sep+'validation_labels_shuffled.npy')


# Build up paths to Images from original dataset
# This would be used in case of large datasets, which a numpy array would not be practical for
train_paths = np.load(r'.'+sep+'data_files'+sep+'training_names_images_shuffled.npy')
test_paths = np.load(r'.'+sep+'data_files'+sep+'testing_names_images_shuffled.npy')
validate_paths = np.load(r'.'+sep+'data_files'+sep+'validation_names_images_shuffled.npy')

mean_std = np.load(r'.'+sep+'data_files'+sep+'std_mean.npy', allow_pickle=True)

lengths = [len(train_labels), len(test_labels), len(validate_labels)]


# Normalize input data
def data_normalization(image, label):
    image = tf.image.resize_images(image, size=(224, 224))
    image = tf.to_float(image)
    mean = [mean_std[0], mean_std[2], mean_std[4]]
    std_dev = [mean_std[1], mean_std[3], mean_std[5]]
    mean = tf.reshape(mean, [1, 1, 3])
    std_dev = tf.reshape(std_dev, [1, 1, 3])
    img_m = image - mean
    image = img_m / std_dev
    return image, label


# Setup the dataset API to load the data
def load_data(batch_size, buffer_size):

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.map(data_normalization, num_parallel_calls=8)
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_dataset = test_dataset.map(data_normalization, num_parallel_calls=8)
    test_dataset = test_dataset.shuffle(buffer_size).batch(batch_size)
    test_dataset = test_dataset.prefetch(buffer_size)

    validation_dataset = tf.data.Dataset.from_tensor_slices((validate_images, validate_labels))
    validation_dataset = validation_dataset.map(data_normalization, num_parallel_calls=8)
    validation_dataset = validation_dataset.shuffle(buffer_size).batch(batch_size)
    validation_dataset = validation_dataset.prefetch(buffer_size)

    return train_dataset, test_dataset, validation_dataset, lengths




