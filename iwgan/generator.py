import numpy as np
import tensorflow as tf

def generator(z, channels=3):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        start_resolution = 7
        start_features = 256
        x = tf.layers.dense(inputs=z, units=start_resolution*start_resolution*start_features, activation=tf.nn.relu)
        x = tf.reshape(x, shape=(-1, start_resolution, start_resolution, start_features))

        x = tf.layers.conv2d_transpose(inputs=x, filters=128, kernel_size=2, strides=1, padding='same') # 7x7
        x = tf.nn.relu(tf.layers.batch_normalization(x))
        
        x = tf.layers.conv2d_transpose(inputs=x, filters=64, kernel_size=2, strides=2, padding='same') # 14x14
        x = tf.nn.relu(tf.layers.batch_normalization(x))

        x = tf.layers.conv2d_transpose(inputs=x, filters=channels, kernel_size=2, strides=2, padding='same') # 28x28
        x = tf.nn.sigmoid(tf.layers.batch_normalization(x))

        return x

def train(gen, lr, beta1, beta2):
    loss = tf.reduce_mean(-1 * gen)
    optimiser = tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2)\
        .minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator"))
    return loss, optimiser