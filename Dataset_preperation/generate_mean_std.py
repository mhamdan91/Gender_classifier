import os
import numpy as np
import time
import sys

'''
This file generates mean and standard deviation for the training dataset

'''


sep = os.sep
HOME = os.getcwd()
MAIN = HOME+sep+r'combined'

# Function to monitor progress
def print_progress(cnt, total_items, time_):
    percent_complete = float(cnt) / total_items
    sec = int(time_ % 60)
    mint = int(time_/60) % 60
    hr = int(time_/3600) % 60
    ETA = int(time_*int(total_items/cnt) - time_)
    ETA_sec = int(ETA % 60)
    ETA_mint = int(ETA/60) % 60
    ETA_hr = int(ETA/3600) % 60
    msg = "\r Time_lapsed (hr:mm:ss) --> {0:02d}:{1:02d}:{2:02d} , Progress:{3:.1%}, completed {4:d} " \
          "out of {5:d} items , ETA -> (hr:mm:ss): {6:02d}:{7:02d}:{8:02d}".format(hr, mint, sec, percent_complete, cnt, total_items,
                                                                                   ETA_hr,ETA_mint, ETA_sec)
    sys.stdout.write(msg)
    sys.stdout.flush()


'''
Generate Standard div and mean for training set
'''

def dataset_mean_std(images, length):
    red_ch_mean = 0
    green_ch_mean = 0
    blue_ch_mean = 0
    red_ch_std = 0
    green_ch_std = 0
    blue_ch_std = 0
    img_mean = 0.0
    img_std = 0.0
    count = 0
    time_st = time.time()
    img_mean_arr = []
    img_std_arr = []
    for i, img in enumerate(images):
        red_ch_mean += np.mean(img[:, :, 0], axis=(0, 1))    # stretch to single 2D and average over 2axes
        green_ch_mean += np.mean(img[:, :, 1], axis=(0, 1))  # per channel work
        blue_ch_mean += np.mean(img[:, :, 2], axis=(0, 1))
        red_ch_std += np.std(img[:, :, 0], axis=(0, 1))
        green_ch_std += np.std(img[:, :, 1], axis=(0, 1))
        blue_ch_std += np.std(img[:, :, 2], axis=(0, 1))

        img_mean += np.mean(img)  # per image work
        img_std += np.std(img)
        img_mean_arr.append(np.mean(img))
        img_std_arr.append(np.std(img))
        count += 1
        time_ = time.time() - time_st
        if i % 100 == 0:
            print_progress(count, length, time_)

    print('\n count=', count)
    red_ch_mean /= count
    green_ch_mean /= count
    blue_ch_mean /= count
    red_ch_std /= count
    green_ch_std /= count
    blue_ch_std /= count
    img_mean /= count
    img_std /= count
    std_mean_data_ = [red_ch_mean, red_ch_std, green_ch_mean, green_ch_std, blue_ch_mean, blue_ch_std, img_mean, img_std, img_mean_arr, img_std_arr]
    np.save('std_mean.npy', std_mean_data_)  # This file is moved to ../Phase_II/data_files
    return std_mean_data_

# The mean should be computed over training only


generate = False
if generate:
    training_data = np.load('training_data_shuffled.npy')  # This file is moved to ../Phase_II/data_files
    std_mean_data = dataset_mean_std(training_data, len(training_data))
