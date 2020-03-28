Classification based on gender and age
--------------------------------------

# For simple use
To predict the class of the given image "female_1_10.jpg" Run classify.py as follows
* **python3 classify.py -t 0**

## Arguments:
* '-b', '--batch_size', default=8, type=int, help='(Batch size. depends on GPU/CPU ram capacity -- default set to: 8
* '-t', '--train_mode', default=0, type=int, help='0: Predict, 1: Train  -- set to default: 0 '
* '-e', '--training_epochs', default=2, type=int, help='-- default set to 2'
* '-i', '--input_path', default='female_1_10.jpg', type=str, help='(Optional, provide path input image in case of predictions or train_mode = 0) -- defualt 
set to: female_1_10.jpg'
* '-k', '--ckpt_path', default='./checkpoints/cp-91.2-0100.ckpt', type=str, help='(Optional, provide path to checkpoint in case of '
                            'predictions or train_mode = 0) -- defualt set to: ./checkpoints/cp-91.2-0100.ckpt'
    
#### Muhammad Hamdan
