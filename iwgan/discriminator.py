import numpy as np
import tensorflow as tf

def discriminator(X, current_lod, alpha, dim=64, kernel_size=5):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        x = X
        x = layer(x, 4, current_lod, alpha, dim) # 32 x 32
        x = layer(x, 3, current_lod, alpha, 2 * dim) # 16 x 16
        x = layer(x, 2, current_lod, alpha, 4 * dim) # 8 x 8
        x = layer(x, 1, current_lod, alpha, 8 * dim) # 4 x 4

        x = tf.layers.conv2d(x, filters=8 * dim, kernel_size=3, activation=tf.nn.leaky_relu, padding='same')
        x = tf.layers.conv2d(x, filters=8 * dim, kernel_size=4, activation=tf.nn.leaky_relu)
        x = tf.reshape(x, shape=(-1, dim * 8 * 1 * 1))
        x = tf.layers.dense(x, units=1)

        return x


def layer(x, lod, current_lod, alpha, features):
    downsample = tf.layers.average_pooling2d(x, pool_size=2, strides=2)
    downsample = tf.layers.conv2d(downsample, features, kernel_size=1, padding='same', activation=tf.nn.leaky_relu)
    x = conv(x, features)
    x = tf.layers.average_pooling2d(x, pool_size=2, strides=2)
    x = tf.cond(lod > current_lod,
        lambda: downsample,
        lambda: tf.cond(tf.equal(lod, current_lod), lambda: downsample + alpha * (x - downsample), lambda: x))
    return x


def conv(x, features):
    x = tf.layers.conv2d(x, features, kernel_size=3, strides=1, padding='same', activation=tf.nn.leaky_relu)
    # x = tf.layers.conv2d(x, features, kernel_size=3, strides=1, padding='same', activation=tf.nn.leaky_relu)
    return x


def train(real_dis, fake_dis, penalty_dis, lmda, penalty_X, lr, beta1, beta2):
    penalty_gradients = tf.gradients(penalty_dis, penalty_X)
    norm = tf.norm(penalty_gradients[0], axis=(1, 2))
    loss = tf.reduce_mean(fake_dis - real_dis + lmda * tf.pow(norm - 1, 2))
    optimiser = tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2)\
        .minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator"))
    return loss, optimiser


