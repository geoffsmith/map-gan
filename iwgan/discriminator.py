import numpy as np
import tensorflow as tf
from .layers import get_weight, pixel_norm

def discriminator(X, current_lod, alpha, dim=64, kernel_size=5):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        x = X
        x = layer(x, 4, current_lod, alpha, dim) # 32 x 32
        x = layer(x, 3, current_lod, alpha, 2 * dim) # 16 x 16
        x = layer(x, 2, current_lod, alpha, 4 * dim) # 8 x 8
        x = layer(x, 1, current_lod, alpha, 8 * dim) # 4 x 4

        with tf.variable_scope(f'last-conv', reuse=tf.AUTO_REUSE):
            x = conv2d(x, kernel=3, features=8 * dim)
            x = tf.nn.leaky_relu(x)
            x = pixel_norm(x)
        with tf.variable_scope(f'last-conv-2', reuse=tf.AUTO_REUSE):
            x = conv2d(x, kernel=4, features=8 * dim, padding='VALID')
            x = tf.nn.leaky_relu(x)
            x = pixel_norm(x)
        with tf.variable_scope(f'dense', reuse=tf.AUTO_REUSE):
            x = tf.reshape(x, shape=(-1, dim * 8 * 1 * 1))
            x = dense(x, units=1)

        return x


def layer(x, lod, current_lod, alpha, features):
    with tf.variable_scope(f'layer-{lod}', reuse=tf.AUTO_REUSE):
        with tf.variable_scope(f'Downsample', reuse=tf.AUTO_REUSE):
            downsample = tf.layers.average_pooling2d(x, pool_size=2, strides=2)
            downsample = conv2d(downsample, 1, features)
            downsample = tf.nn.leaky_relu(downsample)
        with tf.variable_scope(f'Conv', reuse=tf.AUTO_REUSE):
            x = conv(x, features)
            x = tf.layers.average_pooling2d(x, pool_size=2, strides=2)
            x = tf.cond(lod > current_lod,
                lambda: downsample,
                lambda: tf.cond(tf.equal(lod, current_lod), lambda: downsample + alpha * (x - downsample), lambda: x))
            return x


def conv(x, features):
    x = conv2d(x, 3, features)
    x = tf.nn.leaky_relu(x)
    return x


def conv2d(x, kernel, features, padding='SAME'):
    w = get_weight([kernel, kernel, x.shape[-1].value, features], use_wscale=True)
    w = tf.cast(w, x.dtype)
    x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding=padding)
    return x


def dense(x, units):
    with tf.variable_scope(f'weight', reuse=tf.AUTO_REUSE):
        w = get_weight([x.shape[1].value, units], use_wscale=True)
    with tf.variable_scope(f'bias', reuse=tf.AUTO_REUSE):
        b = get_weight([units, 1], use_wscale=True)
    w = tf.cast(w, x.dtype)
    b = tf.cast(b, x.dtype)
    return tf.matmul(x, w) + b


def train(real_dis, fake_dis, penalty_dis, lmda, penalty_X, lr, beta1, beta2):
    penalty_gradients = tf.gradients(penalty_dis, penalty_X)
    norm = tf.norm(penalty_gradients[0], axis=(1, 2))
    loss = tf.reduce_mean(fake_dis - real_dis + lmda * tf.pow(norm - 1, 2))
    optimiser = tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2)\
        .minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator"))
    return loss, optimiser


