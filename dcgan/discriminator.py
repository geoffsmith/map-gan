import numpy as np
import tensorflow as tf

def discriminator(X):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        # print(X)
        # x = tf.reshape(X, shape=(-1, 28, 28, 1))
        conv1 = tf.layers.conv2d(X, 32, kernel_size=3)
        act1 = tf.nn.relu(tf.layers.batch_normalization(conv1))
        pool1 = tf.layers.max_pooling2d(act1, pool_size=2, strides=2, padding='same')

        conv2 = tf.layers.conv2d(pool1, 64, kernel_size=3)
        act2 = tf.nn.relu(tf.layers.batch_normalization(conv2))
        pool2 = tf.layers.max_pooling2d(act2, pool_size=2, strides=2, padding='same')
        
        flatten = tf.reshape(pool2, shape=(-1, 64 * 8 * 8))
        dense = tf.layers.dense(flatten, units=1)

        return tf.nn.sigmoid(dense)


def train(real_dis, fake_dis):
    loss = -1 * tf.reduce_mean(tf.log(real_dis) + tf.log(1 - fake_dis))
    optimiser = tf.train.GradientDescentOptimizer(0.001).minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator"))
    return loss, optimiser


