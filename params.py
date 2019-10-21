# Data directories
dataset_dir           = '../datasets/DS0'
shape_dir             = dataset_dir+'/shapes_test'
velocity_dir          = dataset_dir+'/velocities_test'
pressure_dir          = dataset_dir+'/pressures'
shape_dir_test        = dataset_dir+'/shapes_test'
velocity_dir_test     = dataset_dir+'/velocities_test'
pressure_dir_test     = dataset_dir+'/pressures_test'

input_dir             = shape_dir
sol_dir               = velocity_dir
input_dir_test        = shape_dir_test
sol_dir_test          = velocity_dir_test

# Model directories
model_dir             = './'
model_h5              = model_dir+'model.h5'
model_json            = model_dir+'model.json'

# Image data
img_width             = 1070
img_height            = 786
downscaling           = 5
color                 = 'rgb'

# Dataset data
train_size            = 0.8
valid_size            = 0.1
tests_size            = 0.1

# Network data
n_filters_initial     = 32
kernel_size           = 3
kernel_transpose_size = 2
stride_size           = 2
pool_size             = 2

# Learning data
learning_rate         = 1.0e-4
batch_size            = 128
n_epochs              = 25
network               = 'U_net'

# Device data
use_gpu               = True

# Set CPU or GPU
if (not use_gpu):
    print('Not using GPUs')
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
else:
    print('Using GPUs')

# Reset tf1 behavior
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Set seeds for reproducibility
# WARNING : not exactly reproducible when using GPUs
import os
import keras      as k
import numpy      as np
import random     as rn

np.random.seed(1)
rn.seed(1)
tf.set_random_seed(1)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
sess         = tf.Session(graph=tf.get_default_graph(),
                          config=session_conf)
tf.keras.backend.set_session(sess)
