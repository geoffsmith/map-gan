import numpy as np
import tensorflow as tf

def discriminator(X):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        x = tf.layers.conv2d(X, 32, kernel_size=3, strides=1)
        x = tf.nn.relu(tf.layers.batch_normalization(x))
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='same')

        x = tf.layers.conv2d(x, 64, kernel_size=3, strides=1)
        x = tf.nn.relu(tf.layers.batch_normalization(x))
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='same')
        
        x = tf.reshape(x, shape=(-1, 64 * 8 * 8))
        x = tf.layers.dense(x, units=1)

        return tf.nn.sigmoid(x)


def train(real_dis, fake_dis):
    loss = -1 * tf.reduce_mean(tf.log(real_dis) + tf.log(1 - fake_dis))
    optimiser = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator"))
    return loss, optimiser


