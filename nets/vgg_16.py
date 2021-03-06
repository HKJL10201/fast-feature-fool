from urllib.request import urlretrieve
import tensorflow as tf
from misc.layers import *
import numpy as np
import os

def model(image, weights, biases, keep_prob=1.0):
    #check image dimensions
    assert image.get_shape().as_list()[1:] == [224, 224, 3]
    layers = {}
    with tf.name_scope("conv1"):
        layers['conv1_1'] = conv_layer(image, weights['conv1_1'], biases['conv1_1'])
        layers['conv1_2'] = conv_layer(layers['conv1_1'], weights['conv1_2'], biases['conv1_2'])
        layers['pool1'] = max_pool(layers['conv1_2'], k=2, s=2)

    with tf.name_scope("conv2"):
        layers['conv2_1'] = conv_layer(layers['pool1'], weights['conv2_1'], biases['conv2_1'])
        layers['conv2_2'] = conv_layer(layers['conv2_1'], weights['conv2_2'], biases['conv2_2'])
        layers['pool2'] = max_pool(layers['conv2_2'], k=2, s=2)

    with tf.name_scope("conv3"):
        layers['conv3_1'] = conv_layer(layers['pool2'], weights['conv3_1'], biases['conv3_1'])
        layers['conv3_2'] = conv_layer(layers['conv3_1'], weights['conv3_2'], biases['conv3_2'])
        layers['conv3_3'] = conv_layer(layers['conv3_2'], weights['conv3_3'], biases['conv3_3'])
        layers['pool3'] = max_pool(layers['conv3_3'], k=2, s=2)

    with tf.name_scope("conv4"):
        layers['conv4_1'] = conv_layer(layers['pool3'], weights['conv4_1'], biases['conv4_1'])
        layers['conv4_2'] = conv_layer(layers['conv4_1'], weights['conv4_2'], biases['conv4_2'])
        layers['conv4_3'] = conv_layer(layers['conv4_2'], weights['conv4_3'], biases['conv4_3'])
        layers['pool4'] = max_pool(layers['conv4_3'], k=2, s=2)

    with tf.name_scope("conv5"):
        layers['conv5_1'] = conv_layer(layers['pool4'], weights['conv5_1'], biases['conv5_1'])
        layers['conv5_2'] = conv_layer(layers['conv5_1'], weights['conv5_2'], biases['conv5_2'])
        layers['conv5_3'] = conv_layer(layers['conv5_2'], weights['conv5_3'], biases['conv5_3'])
        layers['pool5'] = max_pool(layers['conv5_3'], k=2, s=2)

    flatten = tf.reshape(layers['pool5'], [-1, 25088])

    with tf.name_scope('fc6'):
        layers['fc6'] = tf.nn.relu(fully_connected(flatten, weights['fc6'], biases['fc6']))
        layers['fc6'] = tf.nn.dropout(layers['fc6'], keep_prob=keep_prob)

    with tf.name_scope('fc7'):
        layers['fc7'] = tf.nn.relu(fully_connected(layers['fc6'], weights['fc7'], biases['fc7']))
        layers['fc7'] = tf.nn.dropout(layers['fc7'], keep_prob=keep_prob)

    with tf.name_scope('fc8'):
        layers['fc8'] = fully_connected(layers['fc7'], weights['fc8'], biases['fc8'])
        layers['prob'] = tf.nn.softmax(layers['fc8'])

    return layers

def vgg16(input):
    #weigths and biases for tensorflow
    weights_path = os.path.join('weights', 'vgg16.npy')
    if not os.path.isfile(weights_path):
        print('Downloading VGG-16 weights...')
        urlretrieve (" https://www.dropbox.com/s/zpeufcwesimhvua/vgg16.npy?raw=1", os.path.join('weights', 'vgg16.npy'))
    net = np.load('weights/vgg16.npy',allow_pickle=True,encoding="latin1").item()
    weights = {}
    biases = {}
    for name in net.keys():
        weights[name] = tf.Variable(tf.constant(net[name][0]), dtype='float32' ,name=name+"_weight", trainable=False)
        biases[name] = tf.Variable(tf.constant(net[name][1]), dtype='float32' ,name=name+"_bias", trainable=False)

    return model(input, weights, biases)
