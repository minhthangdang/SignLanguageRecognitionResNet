import numpy as np
import pandas as pd
import tensorflow as tf
import math
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform

def load_dataset():
	# read train dataset
	train_dataset = pd.read_csv('sign_mnist_train.csv')
	train_set_y_orig = labels = train_dataset['label'].values # train set labels
	train_dataset.drop('label', axis = 1, inplace = True) # drop the label coloumn from the training set
	train_set_x_orig = train_dataset.values # train set features
	# convert X to (m, n_H, n_W, n_C) where 
	# m is number of examples, n_H is height, n_W is width and n_C is number of channels
	train_set_x_orig = train_set_x_orig.reshape((train_set_x_orig.shape[0], 28, 28, 1))

	# read test dataset
	test_dataset = pd.read_csv('sign_mnist_test.csv')
	test_set_y_orig = test_dataset['label'].values # test set labels
	test_dataset.drop('label', axis = 1, inplace = True) # drop the label coloumn from the test set
	test_set_x_orig = test_dataset.values # test set features
	# convert X to (m, n_H, n_W, n_C) where 
	# m is number of examples, n_H is height, n_W is width and n_C is number of channels
	test_set_x_orig = test_set_x_orig.reshape((test_set_x_orig.shape[0], 28, 28, 1))

	classes = np.array(labels)
	classes = np.unique(classes)

	return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def identity_block(X, filters, stage, block):
    """
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (3, 3), strides = (1,1), padding = 'same', name = conv_name_base + '2a', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
       
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (3, 3), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def convolutional_block(X, filters, stage, block):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2 = filters
    
    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (3, 3), strides = (2, 2), padding = 'same', name = conv_name_base + '2a', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(F2, (3, 3), strides = (1, 1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(F2, (3, 3), strides = (2, 2), padding = 'same', name = conv_name_base + '1', kernel_initializer = glorot_uniform())(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X