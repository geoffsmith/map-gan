import numpy as np
import tensorflow as tf

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in) # He init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))
        

def pixel_norm(x, axis=3):
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=axis, keep_dims=True) + 1e-8)

def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[-1]], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return tf.nn.bias_add(x, b, data_format='NHWC')


def dense(x, units):
    with tf.variable_scope(f'weight', reuse=tf.AUTO_REUSE):
        w = get_weight([x.shape[1].value, units], use_wscale=True)
    with tf.variable_scope(f'bias', reuse=tf.AUTO_REUSE):
        b = get_weight([units, 1], use_wscale=False)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)
    

def conv2d(x, kernel, features, padding='SAME', use_wscale=True):
    w = get_weight([kernel, kernel, x.shape[-1].value, features], use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    # tf.summary.histogram('weight', w)
    x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding=padding)
    return x