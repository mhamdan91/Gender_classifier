from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import VGG_16 as VGG

tfe = tf.contrib.eager
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
layers = tf.keras.layers
tf.enable_eager_execution(config=config)
tf.executing_eagerly()
print(tf.__version__)
###############################################
''''
This file calls the VGG model and makes a prediction on the given 
image from the original caffe work

'''

image = cv.imread("ak.png")

print(image.shape)
averageImg = [129.1863, 104.7624, 93.5940]

print("***********************")
# Image pre-processing per the matlab code !!!
b, g, r = cv.split(image)
r = r.astype(np.float32)
g = g.astype(np.float32)
b = b.astype(np.float32)
r = r - averageImg[0]
g = g - averageImg[1]
b = b - averageImg[2]
img = cv.merge((b, g, r))
img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)

plt.imshow(img)
plt.show()
img = np.expand_dims(img, axis=0) #(1,224,224,3)
# img = np.rollaxis(img, 3, 1)  # in case channels should be first
imagez = tf.data.Dataset.from_tensor_slices(img).batch(1)
print(img.shape)
print("***********************")


# instantiate a model
data_format = 'channels_last'
model = VGG.VGG16(classes=2622, data_format=data_format, dropout=False)
tf.global_variables_initializer()
for (batch, (images)) in (enumerate(imagez)):
    predict = model(images)  # predict
    index_prd = tf.argmax(predict, 1)
    print(index_prd, predict)
	#print(model.weights)


