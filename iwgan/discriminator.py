import numpy as np
import tensorflow as tf

def discriminator(X, dim=64, kernel_size=5):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        x = tf.layers.conv2d(X, dim, kernel_size=kernel_size, strides=1, padding='same')
        x = tf.nn.leaky_relu(x)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='same') # 32 x 32

        x = tf.layers.conv2d(x, dim * 2, kernel_size=kernel_size, strides=1, padding='same')
        x = tf.nn.leaky_relu(x)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='same') # 16 x 16

        x = tf.layers.conv2d(x, dim * 4, kernel_size=kernel_size, strides=1, padding='same')
        x = tf.nn.leaky_relu(x)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='same') # 8 x 8
        
        x = tf.reshape(x, shape=(-1, dim * 8 * 8 * 4))
        x = tf.layers.dense(x, units=1)

        return x


def train(real_dis, fake_dis, penalty_dis, lmda, penalty_X, lr, beta1, beta2):
    penalty_gradients = tf.gradients(penalty_dis, penalty_X)
    norm = tf.norm(penalty_gradients[0], axis=(1, 2))
    loss = tf.reduce_mean(fake_dis - real_dis + lmda * tf.pow(norm - 1, 2))
    optimiser = tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2)\
        .minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator"))
    return loss, optimiser


