import numpy as np
import tensorflow as tf

def generator(z, channels=3):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        start_resolution = 7
        start_features = 256
        z_input = tf.layers.dense(inputs=z, units=start_resolution*start_resolution*start_features, activation=tf.nn.relu)
        reshape_z = tf.reshape(z_input, shape=(-1, start_resolution, start_resolution, start_features))


        # conv1 = tf.layers.conv2d_transpose(inputs=reshape_z, filters=128, kernel_size=2, strides=3, padding='same') # 7x7
        # act1 = tf.nn.relu(tf.layers.batch_normalization(conv1))
        
        conv2 = tf.layers.conv2d_transpose(inputs=reshape_z, filters=64, kernel_size=2, strides=2, padding='same') # 14x14
        act2 = tf.nn.relu(tf.layers.batch_normalization(conv2))

        conv3 = tf.layers.conv2d_transpose(inputs=act2, filters=channels, kernel_size=2, strides=2, padding='same') # 28x28
        act3 = tf.nn.sigmoid(tf.layers.batch_normalization(conv3))

        return act3

def train(gen):
    loss = tf.reduce_mean(tf.log(1 - gen))
    optimiser = tf.train.GradientDescentOptimizer(0.001).minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator"))
    return loss, optimiser