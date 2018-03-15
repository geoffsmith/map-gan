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
        
        x = tf.reshape(x, shape=(-1, 64 * 6 * 6))
        x = tf.layers.dense(x, units=1)

        return x


def train(real_dis, fake_dis, penalty_dis, lmda, penalty_X, lr, beta1, beta2):
    penalty_gradients = tf.gradients(penalty_dis, penalty_X)
    norm = tf.norm(penalty_gradients[0], axis=(1, 2))
    loss = tf.reduce_mean(fake_dis - real_dis + lmda * tf.pow(norm - 1, 2))
    optimiser = tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2)\
        .minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator"))
    return loss, optimiser


