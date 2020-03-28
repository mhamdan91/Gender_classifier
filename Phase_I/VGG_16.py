
"""
VGG_16 model definition compatible with TensorFlow's eager execution.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import tensorflow as tf


layers = tf.keras.layers


# pylint: disable=not-callable
class VGG16(tf.keras.Model):

  """
  Instantiates the VGG16 architecture.

  Args:
    classes: number of classes to classify  images into - takes in an integer value
    name: Prefix applied to names of variables created in the model.
    trainable: Is the model trainable? If true, performs backward
        and optimization after call() method.
    dropout: whether to include dropout layers after the fully-connected layers (FC6 and FC7)
    data_format: format for the image. Either 'channels_first' or
      'channels_last'.  'channels_first' is typically faster on GPUs while
      'channels_last' is typically faster on CPUs. See

  Raises:
      ValueError: in case of invalid argument for data_format.
  """

  def __init__(self, classes=1, name='', trainable=False, dropout=True, data_format='channels_last'):    # modified
    super(VGG16, self).__init__(name=name)

    valid_channel_values = ('channels_first', 'channels_last')
    if data_format not in valid_channel_values:
      raise ValueError('Unknown data_format: %s. Valid values: %s' % (data_format, valid_channel_values))
    self.dropout = dropout

    conv1_1_0 = np.load('../out/conv1_1_0.npy', allow_pickle=True)
    conv1_1_1 = np.load('../out/conv1_1_1.npy', allow_pickle=True)
    conv1_2_0 = np.load('../out/conv1_2_0.npy', allow_pickle=True)
    conv1_2_1 = np.load('../out/conv1_2_1.npy', allow_pickle=True)

    conv2_1_0 = np.load('../out/conv2_1_0.npy', allow_pickle=True)
    conv2_1_1 = np.load('../out/conv2_1_1.npy', allow_pickle=True)
    conv2_2_0 = np.load('../out/conv2_2_0.npy', allow_pickle=True)
    conv2_2_1 = np.load('../out/conv2_2_1.npy', allow_pickle=True)

    conv3_1_0 = np.load('../out/conv3_1_0.npy', allow_pickle=True)
    conv3_1_1 = np.load('../out/conv3_1_1.npy', allow_pickle=True)
    conv3_2_0 = np.load('../out/conv3_2_0.npy', allow_pickle=True)
    conv3_2_1 = np.load('../out/conv3_2_1.npy', allow_pickle=True)
    conv3_3_0 = np.load('../out/conv3_3_0.npy', allow_pickle=True)
    conv3_3_1 = np.load('../out/conv3_3_1.npy', allow_pickle=True)

    conv4_1_0 = np.load('../out/conv4_1_0.npy', allow_pickle=True)
    conv4_1_1 = np.load('../out/conv4_1_1.npy', allow_pickle=True)
    conv4_2_0 = np.load('../out/conv4_2_0.npy', allow_pickle=True)
    conv4_2_1 = np.load('../out/conv4_2_1.npy', allow_pickle=True)
    conv4_3_0 = np.load('../out/conv4_3_0.npy', allow_pickle=True)
    conv4_3_1 = np.load('../out/conv4_3_1.npy', allow_pickle=True)

    conv5_1_0 = np.load('../out/conv5_1_0.npy', allow_pickle=True)
    conv5_1_1 = np.load('../out/conv5_1_1.npy', allow_pickle=True)
    conv5_2_0 = np.load('../out/conv5_2_0.npy', allow_pickle=True)
    conv5_2_1 = np.load('../out/conv5_2_1.npy', allow_pickle=True)
    conv5_3_0 = np.load('../out/conv5_3_0.npy', allow_pickle=True)
    conv5_3_1 = np.load('../out/conv5_3_1.npy', allow_pickle=True)

    fc6_0 = np.load('../out/fc6_0.npy', allow_pickle=True)
    fc6_1 = np.load('../out/fc6_1.npy', allow_pickle=True)
    fc7_0 = np.load('../out/fc7_0.npy', allow_pickle=True)
    fc7_1 = np.load('../out/fc7_1.npy', allow_pickle=True)
    fc8_0 = np.load('../out/fc8_0.npy', allow_pickle=True)
    fc8_1 = np.load('../out/fc8_1.npy', allow_pickle=True)


    # Block1
    self.conv1_1 = layers.Conv2D(64, (3, 3), strides=(1, 1), data_format=data_format, padding='same', name='conv1_1', activation='relu',
                               kernel_initializer=tf.constant_initializer(conv1_1_0), bias_initializer=tf.constant_initializer(conv1_1_1), trainable=trainable)
    self.conv1_2 = layers.Conv2D(64, (3, 3), strides=(1, 1), data_format=data_format, padding='same', name='conv1_2', activation='relu',
                               kernel_initializer=tf.constant_initializer(conv1_2_0), bias_initializer=tf.constant_initializer(conv1_2_1), trainable=trainable)
    self.max_pool_1 = layers.MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='pool1')


    # Block2
    self.conv2_1 = layers.Conv2D(128, (3, 3), strides=(1, 1), data_format=data_format, padding='same', name='conv2_1', activation='relu',
                               kernel_initializer= tf.constant_initializer(conv2_1_0), bias_initializer=tf.constant_initializer(conv2_1_1), trainable=trainable)
    self.conv2_2 = layers.Conv2D(128, (3, 3), strides=(1, 1), data_format=data_format, padding='same', name='conv2_2', activation='relu',
                               kernel_initializer= tf.constant_initializer(conv2_2_0), bias_initializer=tf.constant_initializer(conv2_2_1), trainable=trainable)
    self.max_pool_2 = layers.MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='pool2')


    # Block3
    self.conv3_1 = layers.Conv2D(256, (3, 3), strides=(1, 1), data_format=data_format, padding='same', name='conv3_1', activation='relu',
                                 kernel_initializer=tf.constant_initializer(conv3_1_0), bias_initializer=tf.constant_initializer(conv3_1_1), trainable=trainable)
    self.conv3_2 = layers.Conv2D(256, (3, 3), strides=(1, 1), data_format=data_format, padding='same', name='conv3_2', activation='relu',
                                 kernel_initializer=tf.constant_initializer(conv3_2_0), bias_initializer=tf.constant_initializer(conv3_2_1), trainable=trainable)
    self.conv3_3 = layers.Conv2D(256, (3, 3), strides=(1, 1), data_format=data_format, padding='same', name='conv3_3', activation='relu',
                                 kernel_initializer=tf.constant_initializer(conv3_3_0), bias_initializer=tf.constant_initializer(conv3_3_1), trainable=trainable)
    self.max_pool_3 = layers.MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='pool3')


    # Block4
    self.conv4_1 = layers.Conv2D(512, (3, 3), strides=(1, 1), data_format=data_format, padding='same', name='conv4_1', activation='relu',
                                 kernel_initializer=tf.constant_initializer(conv4_1_0), bias_initializer=tf.constant_initializer(conv4_1_1), trainable=trainable)
    self.conv4_2 = layers.Conv2D(512, (3, 3), strides=(1, 1), data_format=data_format, padding='same', name='conv4_2', activation='relu',
                                 kernel_initializer=tf.constant_initializer(conv4_2_0), bias_initializer=tf.constant_initializer(conv4_2_1), trainable=trainable)
    self.conv4_3 = layers.Conv2D(512, (3, 3), strides=(1, 1), data_format=data_format, padding='same', name='conv4_3', activation='relu',
                                 kernel_initializer=tf.constant_initializer(conv4_3_0), bias_initializer=tf.constant_initializer(conv4_3_1), trainable=trainable)
    self.max_pool_4 = layers.MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='pool4')


    # Block5
    self.conv5_1 = layers.Conv2D(512, (3, 3), strides=(1, 1), data_format=data_format, padding='same', name='conv5_1', activation='relu',
                                 kernel_initializer=tf.constant_initializer(conv5_1_0), bias_initializer=tf.constant_initializer(conv5_1_1), trainable=trainable)
    self.conv5_2 = layers.Conv2D(512, (3, 3), strides=(1, 1), data_format=data_format, padding='same', name='conv5_2', activation='relu',
                                 kernel_initializer=tf.constant_initializer(conv5_2_0), bias_initializer=tf.constant_initializer(conv5_2_1), trainable=trainable)
    self.conv5_3 = layers.Conv2D(512, (3, 3), strides=(1, 1), data_format=data_format, padding='same', name='conv5_3', activation='relu',
                                 kernel_initializer=tf.constant_initializer(conv5_3_0), bias_initializer=tf.constant_initializer(conv5_3_1), trainable=trainable)
    self.max_pool_5 = layers.MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format, name='pool5')

    self.flatten = layers.Flatten()
    self.fc6 = layers.Dense(4096, activation="relu", kernel_initializer=tf.constant_initializer(fc6_0), bias_initializer=tf.constant_initializer(fc6_1),
                            name='FC6', trainable=trainable)
    self.dropout_fc6 = layers.Dropout(0.5)

    self.fc7 = layers.Dense(4096, activation="relu", kernel_initializer=tf.constant_initializer(fc7_0), bias_initializer=tf.constant_initializer(fc7_1),
                            name='FC7', trainable=trainable)
    self.dropout_fc7 = layers.Dropout(0.5)

    self.fc8 = layers.Dense(classes, activation="softmax", kernel_initializer=tf.constant_initializer(fc8_0), bias_initializer=tf.constant_initializer(fc8_1),
                            name='FC8', trainable=trainable)

  def call(self, inputs, training=False, mask=None):
    x = self.conv1_1(inputs)
    x = self.conv1_2(x)
    x = self.max_pool_1(x)

    x = self.conv2_1(x)
    x = self.conv2_2(x)
    x = self.max_pool_2(x)

    x = self.conv3_1(x)
    x = self.conv3_2(x)
    x = self.conv3_3(x)
    x = self.max_pool_3(x)

    x = self.conv4_1(x)
    x = self.conv4_2(x)
    x = self.conv4_3(x)
    x = self.max_pool_4(x)

    x = self.conv5_1(x)
    x = self.conv5_2(x)
    x = self.conv5_3(x)
    x = self.max_pool_5(x)

    x = self.flatten(x)

    x = self.fc6(x)

    if self.dropout:
      x = self.dropout_fc6(x)

    x = self.fc7(x)

    if self.dropout:
      x = self.dropout_fc7(x)

    x = self.fc8(x)

    return x
