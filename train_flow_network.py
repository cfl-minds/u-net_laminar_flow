# Generic imports
import time
import numpy as np

# Custom imports
from   params         import *
from   datasets_utils import *
from   networks_utils import *
import networks_utils

# Load images
imgs, n_imgs, height, width, n_channels = load_img_dataset(input_dir,
                                                           downscaling,
                                                           color)

# Load solutions
sols, n_sols, height, width, n_channels = load_img_dataset(sol_dir,
                                                           downscaling,
                                                           color)

# Split data into training, validation and testing sets
(imgs_train,
 imgs_valid,
 imgs_tests) = split_dataset(imgs, train_size, valid_size, tests_size)
(sols_train,
 sols_valid,
 sols_tests) = split_dataset(sols, train_size, valid_size, tests_size)

# Print informations
print('Training   set size is', imgs_train.shape[0])
print('Validation set size is', imgs_valid.shape[0])
print('Test       set size is', imgs_tests.shape[0])
print('Input images downscaled to',str(width)+'x'+str(height))

# Set the network and train it
start              = time.time()
regression         = getattr(networks_utils, network)
model, train_model = regression(imgs_train,
                                sols_train,
                                imgs_valid,
                                sols_valid,
                                imgs_tests,
                                n_filters_initial,
                                kernel_size,
                                kernel_transpose_size,
                                pool_size,
                                stride_size,
                                learning_rate,
                                batch_size,
                                n_epochs,
                                height,
                                width,
                                n_channels)

end = time.time()
print('Training time was ',end-start,' seconds')

# Evaluate score on test set
score = evaluate_model_score(model, imgs_tests, sols_tests)

# Save model
save_keras_model(model)

# Plot accuracy and loss
plot_accuracy_and_loss(train_model)
