import training_loop
import numpy as np
import tensorflow as tf
import argparse
import os
tf.logging.set_verbosity(tf.logging.ERROR)  # disable to see tensorflow warnings


def predictor(Batch_size = 8, train_mode=0, epochs_= 2, input_path='female_1_10.jpg', ckpt_path= './checkpoints/cp-91.2-0100.ckpt'):
    training_loop.train_loop(Batch_size, train_mode, epochs_, input_path, ckpt_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', default=8, type=int, help='(Batch size. depends on GPU/CPU ram capacity -- default set to: 8 ')
    parser.add_argument('-t', '--train_mode', default=0, type=int, help='0: Predict, 1: Train  -- set to default: 0 ')
    parser.add_argument('-e', '--training_epochs', default=2, type=int, help='-- default set to 2')
    parser.add_argument('-i', '--input_path', default='female_1_10.jpg', type=str, help='(Optional, provide path input image in case of predictions or '
                                                                                      'train_mode = 0) -- defualt set to: female_1_10.jpg')
    parser.add_argument('-k', '--ckpt_path', default='./checkpoints/cp-91.2-0100.ckpt', type=str, help='(Optional, provide path to checkpoint in case of '
                            'predictions or train_mode = 0) -- defualt set to: ./checkpoints/cp-91.2-0100.ckpt')
    args = parser.parse_args()
    predictor(args.batch_size, args.train_mode, args.training_epochs, args.input_path, args.ckpt_path)

if __name__ == '__main__':
    main()
