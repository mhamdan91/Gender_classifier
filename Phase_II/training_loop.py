from __future__ import absolute_import, division, print_function
import tensorflow as tf
# from tensorflow.python.eager import tape
import os
import time
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
import sys
tfe = tf.contrib.eager
sep = os.sep

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
layers = tf.keras.layers
tf.enable_eager_execution(config=config)
tf.executing_eagerly()
print(tf.__version__)

###############################################
'''
This file represents the main training loop used to train the VGG model
to predict age and gender. 
'''



import data_loader  # load data_loader .py
import validation  # load validation .py

# Import the network
import VGG_16 as VGG

# Function used to print progress of training
def print_progress(cnt, overall, time_, loss, avg_acc):
    overall_complete = cnt/ float(overall - 1)

    sec = int(time_ % 60)
    mint = int(time_ / 60) % 60
    hr = int(time_ / 3600) % 60
    loss = str(loss)
    msg = "\r Time_lapsed (hr:mm:ss) --> {0:02d}:{1:02d}:{2:02d} ,   Training loss: {3:s}   Avg_accuracy: {4:.1%},     Overall Progress:{5:.1%}," \
          " completed {6:d} out of {7:d} items".format(hr, mint, sec, loss, avg_acc, overall_complete, cnt, overall)
    sys.stdout.write(colored(msg, 'green'))
    sys.stdout.flush()


# Function used to write summaries to tensorboard to visually keep track of training/validation progress
def write_summaries(loss, i, global_step, vars_loc, grads_loc, train=True):
    with summary_writer.as_default():
        with tf.contrib.summary.always_record_summaries():
            if train:
                tf.contrib.summary.scalar("train_loss", loss, step=global_step)
                tf.contrib.summary.scalar("step", i, step=global_step)
                tf.contrib.summary.histogram("weights", vars_loc, step=global_step)
                tf.contrib.summary.histogram("gradients", grads_loc, step=global_step)
            else:
                tf.contrib.summary.scalar("val_loss", loss, step=global_step)


# Instantiate a model, setup optimizer and checkpoints.
class_names = np.load(r'.'+sep+'data_files'+sep+'classes.npy')
data_format = 'channels_last'
num_classes = 14
model = VGG.VGG16(classes=num_classes, trainable=True, data_format=data_format, dropout=False)
optimizer = tf.train.AdamOptimizer()
logdir = 'tensorboard_reporting'+sep
checkpont_path = "checkpoints"+sep+"cp-{ACC:2.1f}-{log:04d}.ckpt"
checkpont_dir = os.path.dirname(checkpont_path)
summary_writer = tf.contrib.summary.create_file_writer(logdir)


def train_loop(batch_size=8, train_mode=0, epochs=2, image_path='female_1_10.jpg', checkpoint_path='./checkpoints/cp-91.2-0100.ckpt'):
    if train_mode == 1:
        Epochs = epochs
        Batch_size = batch_size
        Buffer_size = Batch_size
        acc = 0
        acc_v = 0
        val_step = 0

        tf.global_variables_initializer()

        train_dataset, test_dataset, validation_dataset, lengths = data_loader.load_data(Batch_size, Buffer_size)
        tr_len = lengths[0]
        t_len = lengths[1]
        v_len = lengths[2]
        start_time = time.time()
        acc_temp = 0
        for epoch in range(Epochs):
            for (batch, (images, labels)) in (enumerate(train_dataset)):
                batch += 1
                step = tf.train.get_or_create_global_step()
                # Perform forward and backward passes and monitor gradients
                with tf.GradientTape() as tape:
                    logits_out = model(images)
                    y_prd_cls = tf.argmax(logits_out, dimension=1)
                    one_hot_labels = tf.one_hot(labels, num_classes)
                    equality = tf.equal(y_prd_cls, tf.argmax(one_hot_labels, dimension=1))
                    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels, logits=logits_out)
                    watched_vars = tape.watched_variables()
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step=step)
                loss_metric = np.mean(loss)
                weight_list = model.weights
                acc = (acc + accuracy.numpy())/(batch*1.0)
                time_ = time.time() - start_time

                print_progress(batch*Batch_size, tr_len, time_, loss_metric, acc)
                write_summaries(loss_metric, batch, step, weight_list[0], grads[0], train=True)

            print('\n')
            acc_v, loss_metric_v = validation.validate_model(model, acc_v, v_len, validation_dataset, num_classes, Batch_size)
            write_summaries(loss_metric_v, 0, val_step, 0, 0, train=False)
            val_step += 1
            # Save model parameters
            if acc > acc_temp:
                acc_temp = acc
                model.save_weights(checkpont_path.format(ACC=acc*100, log=epoch))

    else:
        raw_path = checkpoint_path+'.index'
        if os.path.exists(raw_path):
            model.load_weights(checkpoint_path)
        else:
            tf.global_variables_initializer()
        mean_std = np.load(r'.' + sep + 'data_files' + sep + 'std_mean.npy', allow_pickle=True)
        image_ = plt.imread(image_path)
        image = tf.image.resize_images(image_, size=(224, 224))
        image = tf.to_float(image)
        mean = [mean_std[0], mean_std[2], mean_std[4]]
        std_dev = [mean_std[1], mean_std[3], mean_std[5]]
        mean = tf.to_float(tf.reshape(mean, [1, 1, 3]))
        std_dev = tf.to_float(tf.reshape(std_dev, [1, 1, 3]))
        img_m = image - mean
        image = img_m / std_dev
        image = tf.reshape(image, [1, 224, 224, 3])
        image = tf.convert_to_tensor(image)
        logits_out = model(image)
        y_prd_cls = tf.argmax(logits_out, dimension=1)
        string = "Predicted class: " + str(class_names[y_prd_cls])
        print("Predicted class:", class_names[y_prd_cls])
        plt.imshow(image_)
        plt.title(string)
        plt.show()
        # print(model.weights)
