# Import stuff
import sys
import math
import keras

# Additional imports from keras
from keras                      import regularizers
from keras                      import optimizers
from keras.models               import Model
from keras.layers               import Input
from keras.layers               import concatenate
from keras.layers               import Conv2D
from keras.layers               import MaxPooling2D
from keras.layers               import AveragePooling2D
from keras.layers               import Flatten
from keras.layers               import Dense
from keras.layers               import Activation
from keras.layers               import Dropout
from keras.layers               import Conv2DTranspose
from keras.layers               import Lambda
from keras.layers               import BatchNormalization
from keras.layers.convolutional import ZeroPadding2D

# Custom imports
from datasets_utils import *

### ************************************************
### I/O convolutional layer
def io_conv_2D(x,
               filters     = 8,
               kernel_size = 3,
               strides     = 1,
               padding     = 'same',
               activation  = 'relu'):

    x = Conv2D(filters     = filters,
               kernel_size = kernel_size,
               strides     = strides,
               padding     = padding,
               activation  = activation)(x)

    return x

### ************************************************
### I/O max-pooling layer
def io_maxp_2D(x,
               pool_size = 2,
               strides   = 2):

    x = MaxPooling2D(pool_size = pool_size,
                     strides   = strides)(x)

    return x

### ************************************************
### I/O avg-pooling layer
def io_avgp_2D(x,
               pool_size = 2,
               strides   = 2):

    x = AveragePooling2D(pool_size = pool_size,
                        strides   = strides)(x)

    return x

### ************************************************
### I/O convolutional transposed layer
def io_conv_2D_transp(in_layer,
                      n_filters,
                      kernel_size,
                      stride_size):

    out_layer = Conv2DTranspose(filters=n_filters,
                                kernel_size=kernel_size,
                                strides=stride_size,
                                padding='same')(in_layer)

    return out_layer

### ************************************************
### I/O concatenate + zero-pad
def io_concat_pad(in_layer_1,
                  in_layer_2,
                  axis):

    # Compute padding sizes
    shape1_x  = np.asarray(keras.backend.int_shape(in_layer_1)[1])
    shape1_y  = np.asarray(keras.backend.int_shape(in_layer_1)[2])
    shape2_x  = np.asarray(keras.backend.int_shape(in_layer_2)[1])
    shape2_y  = np.asarray(keras.backend.int_shape(in_layer_2)[2])
    dx        = shape2_x - shape1_x
    dy        = shape2_y - shape1_y

    # Pad and concat
    pad_layer = ZeroPadding2D(((dx,0),(dy,0)))(in_layer_1)
    out_layer = concatenate([pad_layer, in_layer_2], axis=axis)

    return out_layer

### ************************************************
### Classic U-net for field prediction
def U_net(train_im,
          train_sol,
          valid_im,
          valid_sol,
          test_im,
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
          n_channels):

    # Generate inputs
    conv0 = Input((height,width,n_channels))

    # 2 convolutions + maxPool
    conv1 = io_conv_2D(conv0, n_filters_initial*(2**0), kernel_size)
    conv1 = io_conv_2D(conv1, n_filters_initial*(2**0), kernel_size)
    pool1 = io_maxp_2D(conv1, pool_size)

    # 2 convolutions + maxPool
    conv2 = io_conv_2D(pool1, n_filters_initial*(2**1), kernel_size)
    conv2 = io_conv_2D(conv2, n_filters_initial*(2**1), kernel_size)
    pool2 = io_maxp_2D(conv2, pool_size)

    # 2 convolutions + maxPool
    conv3 = io_conv_2D(pool2, n_filters_initial*(2**2), kernel_size)
    conv3 = io_conv_2D(conv3, n_filters_initial*(2**2), kernel_size)
    pool3 = io_maxp_2D(conv3, pool_size)

    # 2 convolutions + maxPool
    conv4 = io_conv_2D(pool3, n_filters_initial*(2**3), kernel_size)
    conv4 = io_conv_2D(conv4, n_filters_initial*(2**3), kernel_size)
    pool4 = io_maxp_2D(conv4, pool_size)

    # 2 convolutions
    conv5 = io_conv_2D(pool4, n_filters_initial*(2**4), kernel_size)
    conv5 = io_conv_2D(conv5, n_filters_initial*(2**4), kernel_size)

    pre6  = io_conv_2D_transp(conv5, n_filters_initial*(2**3), (2,2), (2,2))
    # 1 transpose convolution and concat + 2 convolutions
    up6   = io_concat_pad(pre6, conv4, 3)
    conv6 = io_conv_2D(up6,   n_filters_initial*(2**3), kernel_size)
    conv6 = io_conv_2D(conv6, n_filters_initial*(2**3), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre7  = io_conv_2D_transp(conv6, n_filters_initial*(2**2), (2,2), (2,2))
    up7   = io_concat_pad(pre7, conv3, 3)
    conv7 = io_conv_2D(up7,   n_filters_initial*(2**2), kernel_size)
    conv7 = io_conv_2D(conv7, n_filters_initial*(2**2), kernel_size)

    pre8  = io_conv_2D_transp(conv7, n_filters_initial*(2**1), (2,2), (2,2))
    # 1 transpose convolution and concat + 2 convolutions
    up8   = io_concat_pad(pre8, conv2, 3)
    conv8 = io_conv_2D(up8,   n_filters_initial*(2**1), kernel_size)
    conv8 = io_conv_2D(conv8, n_filters_initial*(2**1), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre9  = io_conv_2D_transp(conv8, n_filters_initial*(2**0), (2,2), (2,2))
    up9   = io_concat_pad(pre9, conv1, 3)
    conv9 = io_conv_2D(up9,   n_filters_initial*(2**0), kernel_size)
    conv9 = io_conv_2D(conv9, n_filters_initial*(2**0), kernel_size)

    # final 1x1 convolution
    conv10 = io_conv_2D(conv9, 3, 1)

    # construct model
    model = Model(inputs=[conv0], outputs=[conv10])

    # Print info about model
    model.summary()

    # Set training parameters
    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.Adam(lr=learning_rate),
                  metrics=['mean_absolute_error'])

    # Train network
    train_model = model.fit(train_im, train_sol,
                            batch_size=batch_size, epochs=n_epochs,
                            validation_data=(valid_im, valid_sol))

    return(model, train_model)


### ************************************************
### Stacked U-nets
def StackedU_net(train_im,
                 train_sol,
                 valid_im,
                 valid_sol,
                 test_im,
                 n_filters_initial,
                 kernel_size,
                 kernel_size_2,
                 kernel_transpose_size,
                 pool_size,
                 stride_size,
                 learning_rate,
                 batch_size,
                 n_epochs,
                 height,
                 width,
                 n_channels):

    # Generate inputs
    conv0 = Input((height,width,n_channels))

    # 2 convolutions + maxPool
    conv1 = io_conv_2D(conv0, n_filters_initial*(2**0), kernel_size)
    conv1 = io_conv_2D(conv1, n_filters_initial*(2**0), kernel_size)
    pool1 = io_maxp_2D(conv1, pool_size)

    # 2 convolutions + maxPool
    conv2 = io_conv_2D(pool1, n_filters_initial*(2**1), kernel_size)
    conv2 = io_conv_2D(conv2, n_filters_initial*(2**1), kernel_size)
    pool2 = io_maxp_2D(conv2, pool_size)

    # 2 convolutions + maxPool
    conv3 = io_conv_2D(pool2, n_filters_initial*(2**2), kernel_size)
    conv3 = io_conv_2D(conv3, n_filters_initial*(2**2), kernel_size)
    pool3 = io_maxp_2D(conv3, pool_size)

    # 2 convolutions + maxPool
    conv4 = io_conv_2D(pool3, n_filters_initial*(2**3), kernel_size)
    conv4 = io_conv_2D(conv4, n_filters_initial*(2**3), kernel_size)
    pool4 = io_maxp_2D(conv4, pool_size)

    # 2 convolutions
    conv5 = io_conv_2D(pool4, n_filters_initial*(2**4), kernel_size)
    conv5 = io_conv_2D(conv5, n_filters_initial*(2**4), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre6  = io_conv_2D_transp(conv5, n_filters_initial*(2**3), (2,2), (2,2))
    up6   = io_concat_pad(pre6, conv4, 3)
    conv6 = io_conv_2D(up6,   n_filters_initial*(2**3), kernel_size)
    conv6 = io_conv_2D(conv6, n_filters_initial*(2**3), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre7  = io_conv_2D_transp(conv6, n_filters_initial*(2**2), (2,2), (2,2))
    up7   = io_concat_pad(pre7, conv3, 3)
    conv7 = io_conv_2D(up7,   n_filters_initial*(2**2), kernel_size)
    conv7 = io_conv_2D(conv7, n_filters_initial*(2**2), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre8  = io_conv_2D_transp(conv7, n_filters_initial*(2**1), (2,2), (2,2))
    up8   = io_concat_pad(pre8, conv2, 3)
    conv8 = io_conv_2D(up8,   n_filters_initial*(2**1), kernel_size)
    conv8 = io_conv_2D(conv8, n_filters_initial*(2**1), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre9  = io_conv_2D_transp(conv8, n_filters_initial*(2**0), (2,2), (2,2))
    up9   = io_concat_pad(pre9, conv1, 3)
    conv9 = io_conv_2D(up9,   n_filters_initial*(2**0), kernel_size)
    conv9 = io_conv_2D(conv9, n_filters_initial*(2**0), kernel_size)

    # final 1x1 convolution
    conv10 = io_conv_2D(conv9, 3, 1)

    # 2 convolutions + maxPool
    conv21 = io_conv_2D(conv10, n_filters_initial*(2**0), kernel_size_2)
    conv21 = io_conv_2D(conv21, n_filters_initial*(2**0), kernel_size_2)
    pool21 = io_maxp_2D(conv21, pool_size)

    # 2 convolutions + maxPool
    conv22 = io_conv_2D(pool21, n_filters_initial*(2**1), kernel_size_2)
    conv22 = io_conv_2D(conv22, n_filters_initial*(2**1), kernel_size_2)
    pool22 = io_maxp_2D(conv22, pool_size)


    # 2 convolutions + maxPool
    conv23 = io_conv_2D(pool22, n_filters_initial*(2**2), kernel_size_2)
    conv23 = io_conv_2D(conv23, n_filters_initial*(2**2), kernel_size_2)
    pool23 = io_maxp_2D(conv23, pool_size)

    # 2 convolutions + maxPool
    conv24 = io_conv_2D(pool23, n_filters_initial*(2**3), kernel_size_2)
    conv24 = io_conv_2D(conv24, n_filters_initial*(2**3), kernel_size_2)
    pool24 = io_maxp_2D(conv24, pool_size)

    # 2 convolutions
    conv25 = io_conv_2D(pool24, n_filters_initial*(2**4), kernel_size_2)
    conv25 = io_conv_2D(conv25, n_filters_initial*(2**4), kernel_size_2)

    # 1 transpose convolution and concat + 2 convolutions
    pre26  = io_conv_2D_transp(conv25, n_filters_initial*(2**3), (2,2), (2,2))
    up26   = io_concat_pad(pre26, conv24, 3)
    conv26 = io_conv_2D(up26,   n_filters_initial*(2**3), kernel_size_2)
    conv26 = io_conv_2D(conv26, n_filters_initial*(2**3), kernel_size_2)

    pre27  = io_conv_2D_transp(conv26, n_filters_initial*(2**2), (2,2), (2,2))
    up27   = io_concat_pad(pre27, conv23, 3)
    conv27 = io_conv_2D(up27,   n_filters_initial*(2**2), kernel_size_2)
    conv27 = io_conv_2D(conv27, n_filters_initial*(2**2), kernel_size_2)

    pre28  = io_conv_2D_transp(conv27, n_filters_initial*(2**1), (2,2), (2,2))
    up28   = io_concat_pad(pre28, conv22, 3)
    conv28 = io_conv_2D(up28,   n_filters_initial*(2**1), kernel_size_2)
    conv28 = io_conv_2D(conv28, n_filters_initial*(2**1), kernel_size_2)

    # 1 transpose convolution and concat + 2 convolutions
    pre29  = io_conv_2D_transp(conv28, n_filters_initial*(2**0), (2,2), (2,2))
    up29   = io_concat_pad(pre29, conv21, 3)
    conv29 = io_conv_2D(up29,   n_filters_initial*(2**0), kernel_size_2)
    conv29 = io_conv_2D(conv29, n_filters_initial*(2**0), kernel_size_2)

    # final 1x1 convolution
    conv20 = io_conv_2D(conv29, 3, 1)

    # construct model
    model = Model(inputs=[conv0], outputs=[conv20])

    # Print info about model
    model.summary()

    # Set training parameters
    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.Adam(lr=learning_rate),
                  metrics=['mean_absolute_error'])

    # Train network
    train_model = model.fit(train_im, train_sol,
                            batch_size=batch_size, epochs=n_epochs,
                            validation_data=(valid_im, valid_sol))

    return(model, train_model)

### ************************************************
### Coupled U-nets
def CpU_net(train_im,
            train_sol,
            valid_im,
            valid_sol,
            test_im,
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
            n_channels):

    # Generate inputs
    conv0 = Input((height,width,n_channels))

    # 2 convolutions + maxPool
    conv1 = io_conv_2D(conv0, n_filters_initial*(2**0), kernel_size)
    conv1 = io_conv_2D(conv1, n_filters_initial*(2**0), kernel_size)
    pool1 = io_maxp_2D(conv1, pool_size)

    # 2 convolutions + maxPool
    conv2 = io_conv_2D(pool1, n_filters_initial*(2**1), kernel_size)
    conv2 = io_conv_2D(conv2, n_filters_initial*(2**1), kernel_size)
    pool2 = io_maxp_2D(conv2, pool_size)

    # 2 convolutions + maxPool
    conv3 = io_conv_2D(pool2, n_filters_initial*(2**2), kernel_size)
    conv3 = io_conv_2D(conv3, n_filters_initial*(2**2), kernel_size)
    pool3 = io_maxp_2D(conv3, pool_size)

    # 2 convolutions + maxPool
    conv4 = io_conv_2D(pool3, n_filters_initial*(2**3), kernel_size)
    conv4 = io_conv_2D(conv4, n_filters_initial*(2**3), kernel_size)
    pool4 = io_maxp_2D(conv4, pool_size)

    # 2 convolutions
    conv5 = io_conv_2D(pool4, n_filters_initial*(2**4), kernel_size)
    conv5 = io_conv_2D(conv5, n_filters_initial*(2**4), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre6  = io_conv_2D_transp(conv5, n_filters_initial*(2**3), (2,2), (2,2))
    up6   = io_concat_pad(pre6, conv4, 3)
    conv6 = io_conv_2D(up6,   n_filters_initial*(2**3), kernel_size)
    conv6 = io_conv_2D(conv6, n_filters_initial*(2**3), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre7  = io_conv_2D_transp(conv6, n_filters_initial*(2**2), (2,2), (2,2))
    up7   = io_concat_pad(pre7, conv3, 3)
    conv7 = io_conv_2D(up7,   n_filters_initial*(2**2), kernel_size)
    conv7 = io_conv_2D(conv7, n_filters_initial*(2**2), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre8  = io_conv_2D_transp(conv7, n_filters_initial*(2**1), (2,2), (2,2))
    up8   = io_concat_pad(pre8, conv2, 3)
    conv8 = io_conv_2D(up8,   n_filters_initial*(2**1), kernel_size)
    conv8 = io_conv_2D(conv8, n_filters_initial*(2**1), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre9  = io_conv_2D_transp(conv8, n_filters_initial*(2**0), (2,2), (2,2))
    up9   = io_concat_pad(pre9, conv1, 3)
    conv9 = io_conv_2D(up9,   n_filters_initial*(2**0), kernel_size)
    conv9 = io_conv_2D(conv9, n_filters_initial*(2**0), kernel_size)

    # final 1x1 convolution
    conv10 = io_conv_2D(conv9, 3, 1)
    ##### the output of 1-st U-net

    # 2 convolutions + maxPool
    conv21 = io_conv_2D(concatenate([conv0, conv10], axis=3), n_filters_initial*(2**0), kernel_size)
    conv21 = io_conv_2D(conv21, n_filters_initial*(2**0), kernel_size)
    pool21 = io_maxp_2D(conv21, pool_size)

    # 2 convolutions + maxPool
    conv22 = io_conv_2D(concatenate([pool1, pool21], axis=3), n_filters_initial*(2**1), kernel_size)
    conv22 = io_conv_2D(conv22, n_filters_initial*(2**1), kernel_size)
    pool22 = io_maxp_2D(conv22, pool_size)


    # 2 convolutions + maxPool
    conv23 = io_conv_2D(concatenate([pool2, pool22], axis=3), n_filters_initial*(2**2), kernel_size)
    conv23 = io_conv_2D(conv23, n_filters_initial*(2**2), kernel_size)
    pool23 = io_maxp_2D(conv23, pool_size)


    # 2 convolutions + maxPool
    conv24 = io_conv_2D(concatenate([pool3, pool23], axis=3), n_filters_initial*(2**3), kernel_size)
    conv24 = io_conv_2D(conv24, n_filters_initial*(2**3), kernel_size)
    pool24 = io_maxp_2D(conv24, pool_size)


    # 2 convolutions
    conv25 = io_conv_2D(concatenate([pool4, pool24], axis=3), n_filters_initial*(2**4), kernel_size)
    conv25 = io_conv_2D(conv25, n_filters_initial*(2**4), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre26 = io_conv_2D_transp(concatenate([conv5, conv25], axis=3), n_filters_initial*(2**3), (2,2), (2,2))
    up26   = io_concat_pad(pre26, conv24, 3)
    conv26 = io_conv_2D(up26,   n_filters_initial*(2**3), kernel_size)
    conv26 = io_conv_2D(conv26, n_filters_initial*(2**3), kernel_size)

    pre27 = io_conv_2D_transp(concatenate([conv6, conv26], axis=3), n_filters_initial*(2**2), (2,2), (2,2))
    up27   = io_concat_pad(pre27, conv23, 3)
    conv27 = io_conv_2D(up27,   n_filters_initial*(2**2), kernel_size)
    conv27 = io_conv_2D(conv27, n_filters_initial*(2**2), kernel_size)

    pre28 = io_conv_2D_transp(concatenate([conv7, conv27], axis=3), n_filters_initial*(2**1), (2,2), (2,2))
    up28   = io_concat_pad(pre28, conv22, 3)
    conv28 = io_conv_2D(up28,   n_filters_initial*(2**1), kernel_size)
    conv28 = io_conv_2D(conv28, n_filters_initial*(2**1), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre29 = io_conv_2D_transp(concatenate([conv8, conv28], axis=3), n_filters_initial*(2**0), (2,2), (2,2))
    up29   = io_concat_pad(pre29, conv21, 3)
    conv29 = io_conv_2D(up29,   n_filters_initial*(2**0), kernel_size)
    conv29 = io_conv_2D(conv29, n_filters_initial*(2**0), kernel_size)

    # final 1x1 convolution
    conv20 = io_conv_2D(concatenate([conv9, conv29], axis=3), 3, 1)

    # construct model
    model = Model(inputs=[conv0], outputs=[conv20])

    # Print info about model
    model.summary()

    # Set training parameters
    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.Adam(lr=learning_rate),
                  metrics=['mean_absolute_error'])

    # Train network
    train_model = model.fit(train_im, train_sol,
                            batch_size=batch_size, epochs=n_epochs,
                            validation_data=(valid_im, valid_sol))

    return(model, train_model)

### ************************************************
### Multilevel U-nets
def Multi_level_U_net(train_im,
                      train_sol,
                      valid_im,
                      valid_sol,
                      test_im,
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
                      n_channels):

    # Generate inputs
    conv0 = Input((height,width,n_channels))

    # 2 convolutions + maxPool
    conv1 = io_conv_2D(conv0, n_filters_initial*(2**0), kernel_size)
    conv1 = io_conv_2D(conv1, n_filters_initial*(2**0), kernel_size)
    pool1 = io_maxp_2D(conv1, pool_size)

    # 2 convolutions + maxPool
    conv2 = io_conv_2D(pool1, n_filters_initial*(2**1), kernel_size)
    conv2 = io_conv_2D(conv2, n_filters_initial*(2**1), kernel_size)
    pool2 = io_maxp_2D(conv2, pool_size)

    # 2 convolutions + maxPool
    conv3 = io_conv_2D(pool2, n_filters_initial*(2**2), kernel_size)
    conv3 = io_conv_2D(conv3, n_filters_initial*(2**2), kernel_size)
    ########################################################################################################################
    ##Here is the bottle of mini U-net

    pre4 = io_conv_2D_transp(conv3, n_filters_initial*(2**1), (2,2), (2,2))
    # 1 transpose convolution and concat + 2 convolutions
    up4   = io_concat_pad(pre4, conv2, 3)
    conv4 = io_conv_2D(up4,   n_filters_initial*(2**1), kernel_size)
    conv4 = io_conv_2D(conv4, n_filters_initial*(2**1), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre5 = io_conv_2D_transp(conv4, n_filters_initial*(2**0), (2,2), (2,2))
    up5   = io_concat_pad(pre5, conv1, 3)
    conv5 = io_conv_2D(up5,   n_filters_initial*(2**0), kernel_size)
    conv5 = io_conv_2D(conv5, n_filters_initial*(2**0), kernel_size)

    # output of mini U-net
    conv6 = io_conv_2D(conv5, 3, 1)
    ########################################################################################################################
    pool3 = io_maxp_2D(conv3, pool_size)
    conv24 = io_conv_2D(pool3, n_filters_initial*(2**3), kernel_size)
    conv24 = io_conv_2D(conv24, n_filters_initial*(2**3), kernel_size)
    # Here is the bottleneck of small U-net

    pre25 = io_conv_2D_transp(conv24, n_filters_initial*(2**2), (2,2), (2,2))
    up25   = io_concat_pad(pre25, conv3, 3)
    conv25 = io_conv_2D(up25,   n_filters_initial*(2**2), kernel_size)
    conv25 = io_conv_2D(conv25, n_filters_initial*(2**2), kernel_size)

    pre26 = io_conv_2D_transp(conv25, n_filters_initial*(2**1), (2,2), (2,2))
    up26   = io_concat_pad(pre26, conv2, 3)#an alternate is to concatenate pre26 with conv2
    conv26 = io_conv_2D(up26,   n_filters_initial*(2**1), kernel_size)
    conv26 = io_conv_2D(conv26, n_filters_initial*(2**1), kernel_size)

    pre27 = io_conv_2D_transp(conv26, n_filters_initial*(2**0), (2,2), (2,2))
    up27   = io_concat_pad(pre27, conv1, 3)#an alternate is to concatenate pre26 with conv1
    conv27 = io_conv_2D(up27,   n_filters_initial*(2**0), kernel_size)
    conv27 = io_conv_2D(conv27, n_filters_initial*(2**0), kernel_size)

    # output of small U-net
    conv28 = io_conv_2D(conv27, 3, 1)
    ########################################################################################################################
    pool24 = io_maxp_2D(conv24, pool_size)
    conv35 = io_conv_2D(pool24, n_filters_initial*(2**4), kernel_size)
    conv35 = io_conv_2D(conv35, n_filters_initial*(2**4), kernel_size)
    # Here is the bottleneck of U-net

    pre36 = io_conv_2D_transp(conv35, n_filters_initial*(2**3), (2,2), (2,2))
    up36   = io_concat_pad(pre36, conv24, 3)
    conv36 = io_conv_2D(up36,   n_filters_initial*(2**3), kernel_size)
    conv36 = io_conv_2D(conv36, n_filters_initial*(2**3), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre37 = io_conv_2D_transp(conv36, n_filters_initial*(2**2), (2,2), (2,2))## conv36, conv26?
    up37   = io_concat_pad(pre37, conv3, 3)#an alternate is to concatenate pre37 with conv3
    conv37 = io_conv_2D(up37,   n_filters_initial*(2**2), kernel_size)
    conv37 = io_conv_2D(conv37, n_filters_initial*(2**2), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre38 = io_conv_2D_transp(conv37, n_filters_initial*(2**1), (2,2), (2,2))
    up38   = io_concat_pad(pre38, conv2, 3)#two alternates are to concatenate pre38 with conv2 or conv4
    conv38 = io_conv_2D(up38,   n_filters_initial*(2**1), kernel_size)
    conv38 = io_conv_2D(conv38, n_filters_initial*(2**1), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre39 = io_conv_2D_transp(conv38, n_filters_initial*(2**0), (2,2), (2,2))
    up39   = io_concat_pad(pre39, conv1, 3)#two alternates are concatenate pre39 with conv1 or conv5 or conv27
    conv39 = io_conv_2D(up39,   n_filters_initial*(2**0), kernel_size)
    conv39 = io_conv_2D(conv39, n_filters_initial*(2**0), kernel_size)

    # final 1x1 convolution
    conv30 = io_conv_2D(conv39, 3, 1)
    ########################################################################################################################
    # average the output of three U-nets
    conv10 = keras.layers.Average()([conv6, conv28, conv30])
    #concatenate the output of three U-nets
    #conv10 = io_conv_2D(concatenate([conv6, conv28, conv30], axis=3), 3, 1)
    # construct model
    model = Model(inputs=[conv0], outputs=[conv10])

    # Print info about model
    model.summary()

    # Set training parameters
    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.Adam(lr=learning_rate),
                  metrics=['mean_squared_error'])

    # Train network
    train_model = model.fit(train_im, train_sol,
                            batch_size=batch_size, epochs=n_epochs,
                            validation_data=(valid_im, valid_sol))

    return(model, train_model)

### ************************************************
### Inverse multilevel U-net
def InvMU_net(train_im,
          train_sol,
          valid_im,
          valid_sol,
          test_im,
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
          n_channels):

    # Generate inputs
    conv0 = Input((height,width,n_channels))

    # 2 convolutions + maxPool
    conv1 = io_conv_2D(conv0, n_filters_initial*(2**0), kernel_size)
    conv1 = io_conv_2D(conv1, n_filters_initial*(2**0), kernel_size)
    pool1 = io_maxp_2D(conv1, pool_size)

    # 2 convolutions + maxPool
    conv2 = io_conv_2D(pool1, n_filters_initial*(2**1), kernel_size)
    conv2 = io_conv_2D(conv2, n_filters_initial*(2**1), kernel_size)
    pool2 = io_maxp_2D(conv2, pool_size)

    # 2 convolutions + maxPool
    conv3 = io_conv_2D(pool2, n_filters_initial*(2**2), kernel_size)
    conv3 = io_conv_2D(conv3, n_filters_initial*(2**2), kernel_size)
    pool3 = io_maxp_2D(conv3, pool_size)

    conv24 = io_conv_2D(pool3, n_filters_initial*(2**3), kernel_size)
    conv24 = io_conv_2D(conv24, n_filters_initial*(2**3), kernel_size)
    pool24 = io_maxp_2D(conv24, pool_size)

    conv35 = io_conv_2D(pool24, n_filters_initial * (2 ** 4), kernel_size)
    conv35 = io_conv_2D(conv35, n_filters_initial * (2 ** 4), kernel_size)
    # Here is the bottleneck of U-net

    pre36 = io_conv_2D_transp(conv35, n_filters_initial * (2 ** 3), (2, 2), (2, 2))
    up36 = io_concat_pad(pre36, conv24, 3)
    conv36 = io_conv_2D(up36, n_filters_initial * (2 ** 3), kernel_size)
    conv36 = io_conv_2D(conv36, n_filters_initial * (2 ** 3), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre37 = io_conv_2D_transp(conv36, n_filters_initial * (2 ** 2), (2, 2), (2, 2))  ## conv36, conv26?
    up37 = io_concat_pad(pre37, conv3, 3)  # an alternate is to concatenate pre37 with conv3
    conv37 = io_conv_2D(up37, n_filters_initial * (2 ** 2), kernel_size)
    conv37 = io_conv_2D(conv37, n_filters_initial * (2 ** 2), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre38 = io_conv_2D_transp(conv37, n_filters_initial * (2 ** 1), (2, 2), (2, 2))
    up38 = io_concat_pad(pre38, conv2, 3)  # two alternates are to concatenate pre38 with conv2 or conv4
    conv38 = io_conv_2D(up38, n_filters_initial * (2 ** 1), kernel_size)
    conv38 = io_conv_2D(conv38, n_filters_initial * (2 ** 1), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre39 = io_conv_2D_transp(conv38, n_filters_initial * (2 ** 0), (2, 2), (2, 2))
    up39 = io_concat_pad(pre39, conv1, 3)  # two alternates are concatenate pre39 with conv1 or conv5 or conv27
    conv39 = io_conv_2D(up39, n_filters_initial * (2 ** 0), kernel_size)
    conv39 = io_conv_2D(conv39, n_filters_initial * (2 ** 0), kernel_size)

    # final 1x1 convolution
    conv30 = io_conv_2D(conv39, 3, 1)
    conv30 = Lambda(lambda x: x * 1.5)(conv30)
    ########################################################################################################################
    pre25 = io_conv_2D_transp(conv24, n_filters_initial * (2 ** 2), (2, 2), (2, 2))
    up25 = io_concat_pad(pre25, conv37, 3)
    conv25 = io_conv_2D(up25, n_filters_initial * (2 ** 2), kernel_size)
    conv25 = io_conv_2D(conv25, n_filters_initial * (2 ** 2), kernel_size)

    pre26 = io_conv_2D_transp(conv25, n_filters_initial * (2 ** 1), (2, 2), (2, 2))
    up26 = io_concat_pad(pre26, conv38, 3)
    conv26 = io_conv_2D(up26, n_filters_initial * (2 ** 1), kernel_size)
    conv26 = io_conv_2D(conv26, n_filters_initial * (2 ** 1), kernel_size)

    pre27 = io_conv_2D_transp(conv26, n_filters_initial * (2 ** 0), (2, 2), (2, 2))
    up27 = io_concat_pad(pre27, conv39, 3)  # an alternate is to concatenate pre26 with conv1
    conv27 = io_conv_2D(up27, n_filters_initial * (2 ** 0), kernel_size)
    conv27 = io_conv_2D(conv27, n_filters_initial * (2 ** 0), kernel_size)

    # output of small U-net
    conv28 = io_conv_2D(conv27, 3, 1)
    ########################################################################################################################
    pre4 = io_conv_2D_transp(conv3, n_filters_initial*(2**1), (2,2), (2,2))
    # 1 transpose convolution and concat + 2 convolutions
    up4   = io_concat_pad(pre4, conv38, 3)
    conv4 = io_conv_2D(up4,   n_filters_initial*(2**1), kernel_size)
    conv4 = io_conv_2D(conv4, n_filters_initial*(2**1), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre5 = io_conv_2D_transp(conv4, n_filters_initial*(2**0), (2,2), (2,2))
    up5   = io_concat_pad(pre5, conv39, 3)
    conv5 = io_conv_2D(up5,   n_filters_initial*(2**0), kernel_size)
    conv5 = io_conv_2D(conv5, n_filters_initial*(2**0), kernel_size)

    # output of mini U-net
    conv6 = io_conv_2D(conv5, 3, 1)
    conv6 = Lambda(lambda x: x * 0.5)(conv6)
    ########################################################################################################################

    # average the output of three U-nets
    conv10 = keras.layers.Average()([conv6, conv28, conv30])

    # construct model
    model = Model(inputs=[conv0], outputs=[conv10])

    # Print info about model
    model.summary()

    # Set training parameters
    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.Adam(lr=learning_rate),
                  metrics=['mean_squared_error'])

    # Train network
    train_model = model.fit(train_im, train_sol,
                            batch_size=batch_size, epochs=n_epochs,
                            validation_data=(valid_im, valid_sol))

    return(model, train_model)

### ************************************************
### Parallel U-nets
def Parallel_U_net(train_im,
                   train_sol,
                   valid_im,
                   valid_sol,
                   test_im,
                   n_filters_initial,
                   kernel_size,
                   kernel_size_2,
                   kernel_transpose_size,
                   pool_size,
                   stride_size,
                   learning_rate,
                   batch_size,
                   n_epochs,
                   height,
                   width,
                   n_channels):

    # Generate inputs
    conv0 = Input((height,width,n_channels))

    # 2 convolutions + maxPool
    conv1 = io_conv_2D(conv0, n_filters_initial*(2**0), kernel_size)
    conv1 = io_conv_2D(conv1, n_filters_initial*(2**0), kernel_size)
    pool1 = io_maxp_2D(conv1, pool_size)

    # 2 convolutions + maxPool
    conv2 = io_conv_2D(pool1, n_filters_initial*(2**1), kernel_size)
    conv2 = io_conv_2D(conv2, n_filters_initial*(2**1), kernel_size)
    pool2 = io_maxp_2D(conv2, pool_size)

    # 2 convolutions + maxPool
    conv3 = io_conv_2D(pool2, n_filters_initial*(2**2), kernel_size)
    conv3 = io_conv_2D(conv3, n_filters_initial*(2**2), kernel_size)
    pool3 = io_maxp_2D(conv3, pool_size)

    # 2 convolutions + maxPool
    conv4 = io_conv_2D(pool3, n_filters_initial*(2**3), kernel_size)
    conv4 = io_conv_2D(conv4, n_filters_initial*(2**3), kernel_size)
    pool4 = io_maxp_2D(conv4, pool_size)

    # 2 convolutions
    conv5 = io_conv_2D(pool4, n_filters_initial*(2**4), kernel_size)
    conv5 = io_conv_2D(conv5, n_filters_initial*(2**4), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre6 = io_conv_2D_transp(conv5, n_filters_initial*(2**3), (2,2), (2,2))
    up6   = io_concat_pad(pre6, conv4, 3)
    conv6 = io_conv_2D(up6,   n_filters_initial*(2**3), kernel_size)
    conv6 = io_conv_2D(conv6, n_filters_initial*(2**3), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre7 = io_conv_2D_transp(conv6, n_filters_initial*(2**2), (2,2), (2,2))
    up7   = io_concat_pad(pre7, conv3, 3)
    conv7 = io_conv_2D(up7,   n_filters_initial*(2**2), kernel_size)
    conv7 = io_conv_2D(conv7, n_filters_initial*(2**2), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre8 = io_conv_2D_transp(conv7, n_filters_initial*(2**1), (2,2), (2,2))
    up8   = io_concat_pad(pre8, conv2, 3)
    conv8 = io_conv_2D(up8,   n_filters_initial*(2**1), kernel_size)
    conv8 = io_conv_2D(conv8, n_filters_initial*(2**1), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre9 = io_conv_2D_transp(conv8, n_filters_initial*(2**0), (2,2), (2,2))
    up9   = io_concat_pad(pre9, conv1, 3)
    conv9 = io_conv_2D(up9,   n_filters_initial*(2**0), kernel_size)
    conv9 = io_conv_2D(conv9, n_filters_initial*(2**0), kernel_size)

    # final 1x1 convolution
    conv10 = io_conv_2D(conv9, 3, 1)
    #conv10 = keras.layers.Add()([conv0, conv10])
    ##### the output of 1-st U-net

    # 2 convolutions + maxPool
    conv21 = io_conv_2D(conv0, n_filters_initial*(2**0), kernel_size_2)
    conv21 = io_conv_2D(conv21, n_filters_initial*(2**0), kernel_size_2)
    pool21 = io_maxp_2D(conv21, pool_size)

    # 2 convolutions + maxPool
    conv22 = io_conv_2D(pool21, n_filters_initial*(2**1), kernel_size_2)
    conv22 = io_conv_2D(conv22, n_filters_initial*(2**1), kernel_size_2)
    pool22 = io_maxp_2D(conv22, pool_size)


    # 2 convolutions + maxPool
    conv23 = io_conv_2D(pool22, n_filters_initial*(2**2), kernel_size_2)
    conv23 = io_conv_2D(conv23, n_filters_initial*(2**2), kernel_size_2)
    pool23 = io_maxp_2D(conv23, pool_size)


    # 2 convolutions + maxPool
    conv24 = io_conv_2D(pool23, n_filters_initial*(2**3), kernel_size_2)
    conv24 = io_conv_2D(conv24, n_filters_initial*(2**3), kernel_size_2)
    pool24 = io_maxp_2D(conv24, pool_size)


    # 2 convolutions
    conv25 = io_conv_2D(pool24, n_filters_initial*(2**4), kernel_size_2)
    conv25 = io_conv_2D(conv25, n_filters_initial*(2**4), kernel_size_2)

    # 1 transpose convolution and concat + 2 convolutions
    pre26 = io_conv_2D_transp(conv25, n_filters_initial*(2**3), (2,2), (2,2))
    up26   = io_concat_pad(pre26, conv24, 3)
    conv26 = io_conv_2D(up26,   n_filters_initial*(2**3), kernel_size_2)
    conv26 = io_conv_2D(conv26, n_filters_initial*(2**3), kernel_size_2)

    pre27 = io_conv_2D_transp(conv26, n_filters_initial*(2**2), (2,2), (2,2))
    up27   = io_concat_pad(pre27, conv23, 3)
    conv27 = io_conv_2D(up27,   n_filters_initial*(2**2), kernel_size_2)
    conv27 = io_conv_2D(conv27, n_filters_initial*(2**2), kernel_size_2)

    pre28 = io_conv_2D_transp(conv27, n_filters_initial*(2**1), (2,2), (2,2))
    up28   = io_concat_pad(pre28, conv22, 3)
    conv28 = io_conv_2D(up28,   n_filters_initial*(2**1), kernel_size_2)
    conv28 = io_conv_2D(conv28, n_filters_initial*(2**1), kernel_size_2)

    # 1 transpose convolution and concat + 2 convolutions
    pre29 = io_conv_2D_transp(conv28, n_filters_initial*(2**0), (2,2), (2,2))
    up29   = io_concat_pad(pre29, conv21, 3)
    conv29 = io_conv_2D(up29,   n_filters_initial*(2**0), kernel_size_2)
    conv29 = io_conv_2D(conv29, n_filters_initial*(2**0), kernel_size_2)


    # final 1x1 convolution
    conv20 = io_conv_2D(conv29, 3, 1)
    conv30 = keras.layers.Average()([conv10, conv20])


    # construct model
    model = Model(inputs=[conv0], outputs=[conv30])

    # Print info about model
    model.summary()

    # Set training parameters
    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.Adam(lr=learning_rate),
                  metrics=['mean_squared_error'])

    # Train network
    train_model = model.fit(train_im, train_sol,
                            batch_size=batch_size, epochs=n_epochs,
                            validation_data=(valid_im, valid_sol))

    return(model, train_model)
