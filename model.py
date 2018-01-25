import numpy as np

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, Dropout, BatchNormalization
from keras.optimizers import Adam, Nadam
from keras.models import load_model
from keras.constraints import MaxNorm
from keras import regularizers
from keras import layers

from dummy_data import dummy_data_generator

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate

def n_net_3d(input_shape, output_shape, initial_convolutions_num=3, downsize_filters_factor=1, pool_size=(2, 2, 2), initial_learning_rate=0.00001, dropout=.25, filter_shape=(3,3,3), num_outputs=1, deconvolution=True, regression=True):

    # Convenience variables.
    # For now, we assume that the modalties are ordered by nesting priority.
    output_modalities = output_shape[0]

    # Original input
    inputs = Input(input_shape)

    # Change the space of the input data into something a bit more generalized using consecutive convolutions.
    initial_conv = Conv3D(int(8/downsize_filters_factor), filter_shape, activation='relu', padding='same', data_format='channels_first')(inputs)
    initial_conv = BatchNormalization()(initial_conv)
    if initial_convolutions_num > 1:
        for conv_num in xrange(initial_convolutions_num-1):

            initial_conv = Conv3D(int(8/downsize_filters_factor), filter_shape, activation='relu', padding='same', data_format='channels_first')(initial_conv)
            initial_conv = BatchNormalization()(initial_conv)

    # Cascading U-Nets
    input_list = [initial_conv] * output_modalities
    output_list = [None] * output_modalities
    for modality in xrange(output_modalities):

        for output in output_list:
            if output is not None:
                input_list[modality] = concatenate([input_list[modality], output], axis=1)

        print '\n'
        print 'MODALITY', modality, 'INPUT LIST', input_list[modality]
        print '\n'

        output_list[modality] = u_net_3d(input_shape=input_shape, input_tensor=input_list[modality], downsize_filters_factor=downsize_filters_factor*4, pool_size=(2, 2, 2), initial_learning_rate=initial_learning_rate, dropout=dropout, filter_shape=(3,3,3), num_outputs=1, deconvolution=True, regression=True)

    # Concatenate results
    print output_list
    final_output = output_list[0]
    if len(output_list) > 1:
        for output in output_list[1:]:
            final_output = concatenate([final_output, output], axis=1)

    # Get cost
    if regression:
        act = Activation('relu')(final_output)
        model = Model(inputs=inputs, outputs=act)
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss=msq_loss, metrics=[msq])
    else:
        act = Activation('sigmoid')(final_output)
        model = Model(inputs=inputs, outputs=act)
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def w_net_3d(input_shape, output_shape, initial_convolutions_num=3, downsize_filters_factor=1, pool_size=(2, 2, 2), initial_learning_rate=0.00001, dropout=.25, filter_shape=(3,3,3), num_outputs=1, deconvolution=True, regression=True):

    # Convenience variables.
    # For now, we assume that the modalties are ordered by nesting priority.
    output_modalities = output_shape[0]

    inputs = Input(input_shape)

    conv1 = Conv3D(int(32/downsize_filters_factor), (3, 3, 3), activation='relu', data_format='channels_first',
                   padding='same')(inputs)
    conv1 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu', data_format='channels_first',
                   padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size, data_format='channels_first',)(conv1)

    conv2 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first',
                   padding='same')(pool1)
    conv2 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first',
                   padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size, data_format='channels_first')(conv2)

    conv3 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first',
                   padding='same')(pool2)
    conv3 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first',
                   padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size, data_format='channels_first')(conv3)

    conv4 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first',
                   padding='same')(pool3)
    conv4 = Conv3D(int(512/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first',
                   padding='same')(conv4)

    input_list = [conv4] * output_modalities
    output_list = [None] * output_modalities
    layers_list = [{} for x in xrange(output_modalities)]
    previous_layers_list = [{} for x in xrange(output_modalities)]

    for modality in xrange(output_modalities):

        if modality == 0:
            previous_layers_list[modality] = {'conv1': conv1, 'conv2':conv2, 'conv3':conv3}
        else:
            previous_layers_list[modality] = {'conv1': layers_list[modality-1]['conv7'], 'conv2':layers_list[modality-1]['conv6'], 'conv3':layers_list[modality-1]['conv5']}

        layers_list[modality]['up5'] = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=2, nb_filters=int(512/downsize_filters_factor), image_shape=input_shape[-3:])(conv4)
        layers_list[modality]['up5'] = concatenate([layers_list[modality]['up5'], previous_layers_list[modality]['conv3']], axis=1)
        layers_list[modality]['conv5'] = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu', data_format='channels_first',padding='same')(layers_list[modality]['up5'])
        layers_list[modality]['conv5'] = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first',
                       padding='same')(layers_list[modality]['conv5'])

        layers_list[modality]['up6'] = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=1,
                         nb_filters=int(256/downsize_filters_factor),image_shape=input_shape[-3:])(layers_list[modality]['conv5'])
        layers_list[modality]['up6'] = concatenate([layers_list[modality]['up6'], previous_layers_list[modality]['conv2']], axis=1)
        layers_list[modality]['conv6'] = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first', padding='same')(layers_list[modality]['up6'])
        layers_list[modality]['conv6'] = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first',
                       padding='same')(layers_list[modality]['conv6'])

        layers_list[modality]['up7'] = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=0,
                         nb_filters=int(128/downsize_filters_factor), image_shape=input_shape[-3:])(layers_list[modality]['conv6'])
        layers_list[modality]['up7'] = concatenate([layers_list[modality]['up7'], previous_layers_list[modality]['conv1']], axis=1)
        layers_list[modality]['conv7'] = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first', padding='same')(layers_list[modality]['up7'])
        layers_list[modality]['conv7'] = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first',
                       padding='same')(layers_list[modality]['conv7'])

        output_list[modality] = Conv3D(int(1), (1, 1, 1), data_format='channels_first')(layers_list[modality]['conv7'])

    final_output = output_list[0]
    if len(output_list) > 1:
        for output in output_list[1:]:
            final_output = concatenate([final_output, output], axis=1)

    if regression:
        act = Activation('relu')(final_output)
        model = Model(inputs=inputs, outputs=act)
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss=msq_loss, metrics=[msq])
    else:
        act = Activation('sigmoid')(final_output)
        model = Model(inputs=inputs, outputs=act)
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def split_u_net_3d(input_shape=None, input_tensor=None, downsize_filters_factor=1, pool_size=(2, 2, 2), initial_learning_rate=0.00001, convolutions=4, dropout=.1, filter_shape=(3,3,3), num_outputs=1, deconvolution=True, regression=True, activation='relu', output_shape=None):

    # This is messy, as is the part at the conclusion.
    if input_tensor is None:
        inputs = Input(input_shape)
    else:
        inputs = input_tensor

    left_downsize_filters_factor = downsize_filters_factor*4

    input_modalities = input_shape[0]
    left_arms = [{} for x in xrange(input_modalities)]
    for modality in xrange(input_modalities):

        left_arms[modality]['conv1'] = Conv3D(int(32/left_downsize_filters_factor), (3, 3, 3), activation=activation, data_format='channels_first',
                       padding='same')(Lambda(lambda x: inputs[:,modality,:,:], output_shape=(1,) + input_shape[2:])(inputs))
        left_arms[modality]['conv1'] = Conv3D(int(64/left_downsize_filters_factor), (3, 3, 3), activation=activation, data_format='channels_first',
                       padding='same')(left_arms[modality]['conv1'])

        left_arms[modality]['pool1'] = MaxPooling3D(pool_size=pool_size, data_format='channels_first',)(left_arms[modality]['conv1'])

        left_arms[modality]['conv2'] = Conv3D(int(64/left_downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                       padding='same')(left_arms[modality]['pool1'])
        left_arms[modality]['conv2'] = Conv3D(int(128/left_downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                       padding='same')(left_arms[modality]['conv2'])

        left_arms[modality]['pool2'] = MaxPooling3D(pool_size=pool_size, data_format='channels_first')(left_arms[modality]['conv2'])

        left_arms[modality]['conv3'] = Conv3D(int(128/left_downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                       padding='same')(left_arms[modality]['pool2'])
        left_arms[modality]['conv3'] = Conv3D(int(256/left_downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                       padding='same')(left_arms[modality]['conv3'])

        left_arms[modality]['pool3'] = MaxPooling3D(pool_size=(2,2,1), data_format='channels_first')(left_arms[modality]['conv3'])

        left_arms[modality]['conv4'] = Conv3D(int(256/left_downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                       padding='same')(left_arms[modality]['pool3'])
        left_arms[modality]['conv4'] = Conv3D(int(512/left_downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                       padding='same')(left_arms[modality]['conv4'])

    print left_arms
    conv_list = [None]*4
    for conv_idx, conv in enumerate(conv_list):
        conv_string = 'conv' + str(conv_idx+1)
        conv_list[conv_idx] = left_arms[0][conv_string]
        for modality in left_arms[1:]:
            print 'Concatenating', conv_list[conv_idx], 'to', modality[conv_string], '\n'
            conv_list[conv_idx] = concatenate([conv_list[conv_idx], modality[conv_string]], axis=1)

    conv1, conv2, conv3, conv4 = conv_list

    up5 = get_upconv(pool_size=(2,2,1), deconvolution=deconvolution, depth=2, nb_filters=int(512/downsize_filters_factor), image_shape=input_shape[-3:])(conv4)
    up5 = concatenate([up5, conv3], axis=1)
    conv5 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation=activation, data_format='channels_first',padding='same')(up5)
    conv5 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                   padding='same')(conv5)
    # conv5 = BatchNormalization()(conv5)  

    up6 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=1,
                     nb_filters=int(256/downsize_filters_factor),image_shape=input_shape[-3:])(conv5)
    up6 = concatenate([up6, conv2], axis=1)
    conv6 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first', padding='same')(up6)
    conv6 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                   padding='same')(conv6)
    # conv6 = BatchNormalization()(conv6)  

    up7 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=0,
                     nb_filters=int(128/downsize_filters_factor), image_shape=input_shape[-3:])(conv6)
    up7 = concatenate([up7, conv1], axis=1)
    conv7 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first', padding='same')(up7)
    conv7 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                   padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    conv8 = Conv3D(int(num_outputs), (1, 1, 1), data_format='channels_first',)(conv7)

    # Messy
    if input_tensor is not None:
        return conv8

    if regression:
        act = Activation('relu')(conv8)
        model = Model(inputs=inputs, outputs=act)
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss=msq_loss, metrics=[dice_coef])
    else:
        act = Activation('sigmoid')(conv8)
        model = Model(inputs=inputs, outputs=act)
        model.compile(optimizer=Nadam(lr=initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def parellel_unet_3d(input_shape=None, input_tensor=None, downsize_filters_factor=1, pool_size=(2, 2, 2), initial_learning_rate=0.00001, convolutions=4, dropout=.1, filter_shape=(3,3,3), num_outputs=1, deconvolution=True, regression=True, activation='relu', output_shape=None):


    # This is messy, as is the part at the conclusion.
    if input_tensor is None:
        inputs = Input(input_shape)
    else:
        inputs = input_tensor

    left_downsize_filters_factor = downsize_filters_factor*8

    input_modalities = input_shape[0]
    left_arms = [{} for x in xrange(input_modalities)]
    for modality in xrange(input_modalities):

        left_arms[modality]['conv1'] = Conv3D(int(32/left_downsize_filters_factor), (3, 3, 3), activation=activation, data_format='channels_first',
                       padding='same')(inputs)
        left_arms[modality]['conv1'] = Conv3D(int(64/left_downsize_filters_factor), (3, 3, 3), activation=activation, data_format='channels_first',
                       padding='same')(left_arms[modality]['conv1'])

        left_arms[modality]['pool1'] = MaxPooling3D(pool_size=pool_size, data_format='channels_first',)(left_arms[modality]['conv1'])

        left_arms[modality]['conv2'] = Conv3D(int(64/left_downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                       padding='same')(left_arms[modality]['pool1'])
        left_arms[modality]['conv2'] = Conv3D(int(128/left_downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                       padding='same')(left_arms[modality]['conv2'])

        left_arms[modality]['pool2'] = MaxPooling3D(pool_size=pool_size, data_format='channels_first')(left_arms[modality]['conv2'])

        left_arms[modality]['conv3'] = Conv3D(int(128/left_downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                       padding='same')(left_arms[modality]['pool2'])
        left_arms[modality]['conv3'] = Conv3D(int(256/left_downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                       padding='same')(left_arms[modality]['conv3'])

        left_arms[modality]['pool3'] = MaxPooling3D(pool_size=(2,2,1), data_format='channels_first')(left_arms[modality]['conv3'])

        left_arms[modality]['conv4'] = Conv3D(int(256/left_downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                       padding='same')(left_arms[modality]['pool3'])
        left_arms[modality]['conv4'] = Conv3D(int(512/left_downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                       padding='same')(left_arms[modality]['conv4'])

        left_arms[modality]['up5'] = get_upconv(pool_size=(2,2,1), deconvolution=deconvolution, depth=2, nb_filters=int(512/downsize_filters_factor), image_shape=input_shape[-3:])(left_arms[modality]['conv4'])
        left_arms[modality]['up5'] = concatenate([left_arms[modality]['up5'], left_arms[modality]['conv3']], axis=1)
        left_arms[modality]['conv5'] = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation=activation, data_format='channels_first',padding='same')(left_arms[modality]['up5'])
        left_arms[modality]['conv5'] = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                       padding='same')(left_arms[modality]['conv5'])

        left_arms[modality]['up6'] = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=1,
                         nb_filters=int(256/downsize_filters_factor),image_shape=input_shape[-3:])(left_arms[modality]['conv5'])
        left_arms[modality]['up6'] = concatenate([left_arms[modality]['up6'], left_arms[modality]['conv2']], axis=1)
        left_arms[modality]['conv6'] = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first', padding='same')(left_arms[modality]['up6'])
        left_arms[modality]['conv6'] = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                       padding='same')(left_arms[modality]['conv6'])

        left_arms[modality]['up7'] = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=0,
                         nb_filters=int(128/downsize_filters_factor), image_shape=input_shape[-3:])(left_arms[modality]['conv6'])
        left_arms[modality]['up7'] = concatenate([left_arms[modality]['up7'], left_arms[modality]['conv1']], axis=1)
        left_arms[modality]['conv7'] = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first', padding='same')(left_arms[modality]['up7'])
        left_arms[modality]['conv7'] = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                       padding='same')(left_arms[modality]['conv7'])

        left_arms[modality]['conv8'] = Conv3D(int(num_outputs), (1, 1, 1), data_format='channels_first',)(left_arms[modality]['conv7'])

    conv8 = left_arms[0]['conv8']
    for modality in left_arms[1:]:
        conv8 = concatenate([conv8, modality['conv8']], axis=1)

    conv8 = Conv3D(64, (1, 1, 1), data_format='channels_first',)(conv8)
    conv8 = Conv3D(int(num_outputs), (1, 1, 1), data_format='channels_first',)(conv8)

    # Messy
    if input_tensor is not None:
        return conv8

    if regression:
        act = Activation('relu')(conv8)
        model = Model(inputs=inputs, outputs=act)
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss=msq_loss, metrics=[dice_coef])
    else:
        act = Activation('sigmoid')(conv8)
        model = Model(inputs=inputs, outputs=act)
        model.compile(optimizer=Nadam(lr=initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def u_net_3d(input_shape=None, input_tensor=None, downsize_filters_factor=1, pool_size=(2, 2, 2), initial_learning_rate=0.00001, convolutions=4, dropout=.1, filter_shape=(3,3,3), num_outputs=1, deconvolution=True, regression=True, activation='relu', output_shape=None):

    # This is messy, as is the part at the conclusion.
    if input_tensor is None:
        inputs = Input(input_shape)
    else:
        inputs = input_tensor

    conv1 = Conv3D(int(32/downsize_filters_factor), (3, 3, 3), activation=activation, data_format='channels_first',
                   padding='same')(inputs)
    conv1 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation=activation, data_format='channels_first',
                   padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)                   
    pool1 = MaxPooling3D(pool_size=pool_size, data_format='channels_first',)(conv1)

    conv2 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                   padding='same')(pool1)
    conv2 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                   padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)  
    pool2 = MaxPooling3D(pool_size=pool_size, data_format='channels_first')(conv2)

    conv3 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                   padding='same')(pool2)
    conv3 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                   padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)  
    pool3 = MaxPooling3D(pool_size=pool_size, data_format='channels_first')(conv3)

    conv4 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                   padding='same')(pool3)
    conv4 = Conv3D(int(512/downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                   padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)  

    up5 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=2, nb_filters=int(512/downsize_filters_factor), image_shape=input_shape[-3:])(conv4)
    up5 = concatenate([up5, conv3], axis=1)
    conv5 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation=activation, data_format='channels_first',padding='same')(up5)
    conv5 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                   padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)  

    up6 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=1,
                     nb_filters=int(256/downsize_filters_factor),image_shape=input_shape[-3:])(conv5)
    up6 = concatenate([up6, conv2], axis=1)
    conv6 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first', padding='same')(up6)
    conv6 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                   padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)  

    up7 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=0,
                     nb_filters=int(128/downsize_filters_factor), image_shape=input_shape[-3:])(conv6)
    up7 = concatenate([up7, conv1], axis=1)
    conv7 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first', padding='same')(up7)
    conv7 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation=activation,data_format='channels_first',
                   padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    conv8 = Conv3D(int(num_outputs), (1, 1, 1), data_format='channels_first',)(conv7)

    # Messy
    if input_tensor is not None:
        return conv8

    if regression:
        # act = Activation('relu')(conv8)
        model = Model(inputs=inputs, outputs=act)
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss=msq_loss, metrics=[msq_loss])
    else:
        if num_outputs == 1:
            act = Activation('sigmoid')(conv8)
            model = Model(inputs=inputs, outputs=act)
            model.compile(optimizer=Nadam(lr=initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
        else:
            act = Activation(image_softmax)(conv8)  # custom softmax for 4D tensor
            model = Model(inputs=inputs, outputs=act)
            model.compile(optimizer=Nadam(lr=initial_learning_rate), loss=image_categorical_crossentropy_loss,  # custom loss for 4D tensor
                          metrics=[image_categorical_crossentropy])

    return model

def linear_net(input_shape, downsize_filters_factor=1, pool_size=(2, 2, 2), initial_learning_rate=0.00001, convolutions=4, dropout=.25, filter_shape=(3,3,3), num_outputs=1, deconvolution=True, regression=True):

    inputs = Input((1,32,32,32,4))
    conv_mid = Conv3D(int(32/downsize_filters_factor), filter_shape, activation='relu', padding='same', data_format='channels_first')(inputs)

    conv_mid = Dropout(0.25)(conv_mid)

    for conv_num in xrange(convolutions-2):

        conv_mid = Conv3D(int(32/downsize_filters_factor), filter_shape, activation='relu', padding='same', data_format='channels_first')(conv_mid)
        conv_mid = Dropout(0.25)(conv_mid)

    conv_out = Conv3D(int(1), filter_shape, activation='tanh', padding='same', data_format='channels_first', kernel_regularizer=regularizers.l2(0.01))(conv_mid)

    model = Model(inputs=inputs, outputs=conv_out)

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=msq_loss, metrics=[msq])

def vox_net(input_shape=None, input_tensor=None, downsize_filters_factor=1, pool_size=(2, 2, 2), initial_learning_rate=0.00001, convolutions=4, dropout=.25, filter_shape=(3,3,3), num_outputs=1, deconvolution=True, regression=True, output_shape=None):

    def residual_block(residual_tensor):
        # res1 = BatchNormalization()(residual_tensor)
        res1 = Dropout(0.25)(residual_tensor)
        res1 = Conv3D(int(64/downsize_filters_factor), 3, padding='same', kernel_constraint=MaxNorm(2.), strides=(1,1,1), activation='relu', data_format='channels_first')(res1)

        # res2 = BatchNormalization()(res1)
        res2 = Dropout(0.25)(res1)
        res2 = Conv3D(int(64/downsize_filters_factor), 3, padding='same', kernel_constraint=MaxNorm(2.), strides=(1,1,1), activation='relu', data_format='channels_first')(res2)

        res3 = layers.add([res2, residual_tensor])
        return res3

    inputs = Input(input_shape)

    conv1 = Conv3D(int(32/downsize_filters_factor), 3, padding='same', kernel_constraint=MaxNorm(2.), strides=(1,1,1), data_format='channels_first')(inputs)
    
    # conv2 = BatchNormalization()(conv1)
    conv2 = Dropout(0.25)(conv1)
    conv2 = Conv3D(int(64/downsize_filters_factor), 3, padding='same', kernel_constraint=MaxNorm(2.), strides=(1,1,1), activation='relu', data_format='channels_first')(conv2)
    
    # conv3 = BatchNormalization()(conv2)
    conv3 = Dropout(0.25)(conv2)
    conv3 = Conv3D(int(64/downsize_filters_factor), 3, padding='same', kernel_constraint=MaxNorm(2.), strides=(2,2,2), activation='relu', data_format='channels_first')(conv3)
    conv3 = residual_block(conv3)
    conv3 = residual_block(conv3)

    # conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(0.25)(conv3)
    conv4 = Conv3D(int(64/downsize_filters_factor), 3, padding='same', kernel_constraint=MaxNorm(2.), strides=(2,2,2), activation='relu', data_format='channels_first')(conv4)
    conv4 = residual_block(conv4)
    conv4 = residual_block(conv4)

    # conv5 = BatchNormalization()(conv4)
    conv5 = Dropout(0.25)(conv4)
    conv5 = Conv3D(int(64/downsize_filters_factor), 3, padding='same', kernel_constraint=MaxNorm(2.), strides=(2,2,2), activation='relu', data_format='channels_first')(conv5)
    conv5 = residual_block(conv5)
    conv5 = residual_block(conv5)

    upx3 = UpSampling3D(size=(2, 2, 2), data_format='channels_first')(conv3)
    upx4 = UpSampling3D(size=(4, 4, 4), data_format='channels_first')(conv4)
    upx5 = UpSampling3D(size=(8, 8, 8), data_format='channels_first')(conv5)

    upx = layers.add([conv2, upx3, upx4, upx5])

    conv_last = Conv3D(int(num_outputs), (1, 1, 1), data_format='channels_first')(upx)

    if regression:
        act = Activation('relu')(conv_last)
        model = Model(inputs=inputs, outputs=act)
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss=msq_loss, metrics=[msq])
    else:
        act = Activation('sigmoid')(conv_last)
        model = Model(inputs=inputs, outputs=act)
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def msq(y_true, y_pred):
    return K.sum(K.pow(y_true - y_pred, 2), axis=None)

def msq_loss(y_true, y_pred):
    return msq(y_true, y_pred)

# https://stackoverflow.com/questions/43033436/how-to-do-point-wise-categorical-crossentropy-loss-in-keras
def image_softmax(input):  # apply softmax activation to a 4D tensor
    label_dim = 1
    d = K.exp(input - K.max(input, axis=label_dim, keepdims=True))
    return d / K.sum(d, axis=label_dim, keepdims=True)

def image_categorical_crossentropy(y_true, y_pred):  # compute cross-entropy on 4D tensors
    y_pred = K.clip(y_pred,  1e-5, 1 -  1e-5)
    return -K.mean(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))

def image_categorical_crossentropy_loss(y_true, y_pred):  # compute cross-entropy on 4D tensors
    return 1 - image_categorical_crossentropy(y_true, y_pred)

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def neg_dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_f_norm = (y_true_f * 2) - 1
    y_pred_f_norm = (y_pred_f * 2) - 1
    intersection = K.sum(y_true_f_norm * y_pred_f_norm)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return (1 - dice_coef(y_true, y_pred))

def neg_dice_coef_loss(y_true, y_pred):
    return -neg_dice_coef(y_true, y_pred)

def tensorflow_dice_coef(y_true, y_pred, smooth=1):
    # y_pred_u, y_true_u = tf.unstack(y_pred, axis=1), tf.unstack(y_true, axis=1)
    # print y_true_u.get_shape()

    # def dice_calc(y_true, y_pred):
    # intersection = tf.reduce_sum(y_pred * y_true)
    # union =  tf.reduce_sum(y_pred) + tf.reduce_sum(y_true)
    # dice = (2 * intersection + smooth) / (union + smooth)

    # return dice

    x = tf.map_fn(dice_calc, (y_true, y_pred))

    return dice_calc(y_true, y_pred)

def dice_calc(y_true, smooth=1):
    
    y_true = y_true[0]
    y_pred = y_true[0]
    intersection = tf.reduce_sum(y_pred * y_true)
    union =  tf.reduce_sum(y_pred) + tf.reduce_sum(y_true)
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice

def tensorflow_dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def load_old_model(model_file):
    print("Loading pre-trained model")

    # custom_objects = {'msq': msq, 'msq_loss': msq_loss}
    custom_objects = {'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef, 'msq': msq, 'msq_loss': msq_loss, 'tensorflow_dice_coef': tensorflow_dice_coef, 'tensorflow_dice_loss': tensorflow_dice_loss, 'neg_dice_coef_loss': neg_dice_coef_loss, 'neg_dice_coef': neg_dice_coef}

    try:
        from keras_contrib.layers import Deconvolution3D
        custom_objects["Deconvolution3D"] = Deconvolution3D
    except ImportError:
        print("Could not import Deconvolution3D. To use Deconvolution3D install keras-contrib.")

    return load_model(model_file, custom_objects=custom_objects)

def compute_level_output_shape(filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    if depth != 0:
        output_image_shape = np.divide(image_shape, np.multiply(pool_size, depth)).tolist()
    else:
        output_image_shape = image_shape
    return tuple([None, filters] + [int(x) for x in output_image_shape] )

def get_upconv(depth, nb_filters, pool_size, image_shape, kernel_size=(2, 2, 2), strides=(2, 2, 2),
               deconvolution=False):
    if deconvolution and False:
        try:
            from keras_contrib.layers import Deconvolution3D
        except ImportError:
            raise ImportError("Install keras_contrib in order to use deconvolution. Otherwise set deconvolution=False.")

        return Deconvolution3D(filters=nb_filters, kernel_size=kernel_size,
                               output_shape=compute_level_output_shape(filters=nb_filters, depth=depth,
                                                                       pool_size=pool_size, image_shape=image_shape),
                               strides=strides, input_shape=compute_level_output_shape(filters=nb_filters,
                                                                                       depth=depth+1,
                                                                                       pool_size=pool_size,
                                                                                       image_shape=image_shape))
    else:
        return UpSampling3D(size=pool_size, data_format='channels_first')

if __name__ == '__main__':
    model = u_net_3d(input_shape=(4,16,16,16), input_tensor=None, downsize_filters_factor=1, pool_size=(2, 2, 2), initial_learning_rate=0.00001, convolutions=4, dropout=.25, filter_shape=(3,3,3), num_outputs=1, deconvolution=True, regression=False, output_shape=None)
    pass