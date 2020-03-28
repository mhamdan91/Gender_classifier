import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

'''
Data is split into 14 classes based on Age and Gender
Gender is split between male and female -> makes 2 categories
Age is grouped in decades from 1 through above 60 -> makes 7 categories


Data has been generated and moved to ../Phase_II/data_files
'''



classes = ['Female_between_1_10', 'Male_between_1_10',
           'Female_between_11_20', 'Male_between_11_20',
           'Female_between_21_30', 'Male_between_21_30',
           'Female_between_31_40', 'Male_between_31_40',
           'Female_between_41_50', 'Male_between_41_50',
           'Female_between_51_60', 'Male_between_51_60',
           'Female_between_61_70', 'Male_between_61_70']
np.save('classes.npy', classes)

sep = os.sep
HOME = os.getcwd()
MAIN = HOME+sep+r'combined'

train_val = os.listdir(MAIN)
train_data = []
images_path = []
train_labels = []
test_labels = []
test_data = []
labels = []
train_names = []
test_names = []


for i, _ in enumerate(train_val):
    labels.clear()
    images_path.clear()
    data_path = MAIN + sep + _
    rdata = sorted(os.listdir(data_path))
    for folder in tqdm(rdata):
        age = int(folder.split('_')[0])
        gender = folder.split('_')[1]
        imgz_path = os.path.join(data_path, folder)
        images = sorted(os.listdir(imgz_path))
        for image in images:
            img_path = os.path.join(imgz_path, image)
            images_path.append(img_path)
            readIMG = plt.imread(img_path)
            if i == 0:
                train_data.append(readIMG)
            else:
                test_data.append(readIMG)

            if age<11:
                if gender == 'F':
                    labels.append(0)
                else:
                    labels.append(1)
            elif 10 < age < 21:
                if gender == 'F':
                    labels.append(2)
                else:
                    labels.append(3)
            elif 20 < age < 31:
                if gender == 'F':
                    labels.append(4)
                else:
                    labels.append(5)
            elif 30 < age < 41:
                if gender == 'F':
                    labels.append(6)
                else:
                    labels.append(7)
            elif 40 < age < 51:
                if gender == 'F':
                    labels.append(8)
                else:
                    labels.append(9)
            elif 50 < age < 61:
                if gender == 'F':
                    labels.append(10)
                else:
                    labels.append(11)
            else:
                if gender == 'F':
                    labels.append(12)
                else:
                    labels.append(13)
    if i==0:
        train_labels = np.asarray(labels)
        train_data = np.asarray(train_data)
        np.save('training_data.npy', train_data)
        np.save('training_labels.npy', train_labels)
        np.save('training_names.npy', images_path)
    else:
        test_labels = np.asarray(labels)
        test_data = np.asarray(test_data)
        np.save('testing_data.npy', test_data)
        np.save('testing_labels.npy', test_labels)
        np.save('testing_names.npy', images_path)
