# VALIDATION CODE
import tensorflow as tf
import time
import numpy as np
import sys
from termcolor import colored

'''
This is used to validate the model on validation data

'''
###############################################


# Function to monitor progress
def validation_progress(cnt, v_len, time_, loss, accuracy_loc):
    overall_complete = cnt / float(v_len)
    sec = int(time_) % 60
    mint = int(time_ / 60) % 60
    hr = int(time_ / 3600) % 60
    loss = str(loss)
    msg = "\r Validation_Time (hr:mm:ss) --> {0:02d}:{1:02d}:{2:02d} ,   Validation loss: {3:s}   Avg_accuracy: {4:.1%}   Overall Progress:{5:.1%}," \
          " completed {6:d} out of {7:d} logs".format(hr, mint, sec, loss, accuracy_loc, overall_complete, cnt, v_len)
    sys.stdout.write(colored(msg, 'blue'))
    sys.stdout.flush()


'''
Args:
    model_loc : DNN (VGG) model used to validate data on
    acc_v : validation accuracy
    v_len : length of validation data
    dataset: validation dataset object
    num_classes: number of classes to classify to
    batch_size: batch size used to consume data from the dataset API
'''


def validate_model(model_loc, acc_v, v_len, data_set, num_classes, batch_size):
    cnt = 1
    start = time.time()
    loss_metric = 0
    for (batch, (images, labels)) in (enumerate(data_set)):
        batch+=1
        logits_out = model_loc(images)
        y_prd_cls = tf.argmax(logits_out, dimension=1)
        one_hot_labels = tf.one_hot(labels, num_classes)
        equality = tf.equal(y_prd_cls, tf.argmax(one_hot_labels, dimension=1))
        accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels, logits=logits_out)
        loss_metric = np.mean(loss)
        acc_v = (acc_v + accuracy.numpy())/(batch*batch_size*1.0)

        time_ = time.time() - start
        validation_progress(cnt*batch_size, v_len, time_, loss_metric, acc_v)

        cnt += 1
    print('\n')
    return acc_v, loss_metric
