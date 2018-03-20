import numpy as np
import tensorflow as tf
from .layers import get_weight, pixel_norm

def generator(z, lod, alpha, channels=3):
    dim = 32
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        x = tf.reshape(z, shape=(-1, 1, 1, 8 * dim))
        x, out = layer(x, lod, alpha, 0, 8 * dim, channels)      # 8x8
        print('out', out.shape)
        print('x', x.shape)
        x, out = layer(x, lod, alpha, 1, 4 * dim, channels, out) # 16 x 16
        print('out', out.shape)
        print('x', x.shape)
        x, out = layer(x, lod, alpha, 2, 2 * dim, channels, out) # 32 x 32
        print('out', out.shape)
        print('x', x.shape)
        x, out = layer(x, lod, alpha, 3, 1 * dim, channels, out) # 64 x 64
        print('out', out.shape)
        print('x', x.shape)
        _, out = layer(x, lod, alpha, 4, 1 * dim, channels, out) # 64 x 64
        print('out', out.shape)
        out = tf.nn.sigmoid(out)
        print('out', out.shape)

        return out


def layer(x, current_lod, alpha, layer_lod, filters, channels, prev_rgb=None):
    with tf.variable_scope(f'lod-{layer_lod}', reuse=tf.AUTO_REUSE):
        if layer_lod == 0:
            x = conv2d_transpose(x, kernel=4, strides=4, features=filters)
            x = tf.nn.leaky_relu(x)
            x = pixel_norm(x)
            out = to_rgb(x, channels)
            return x, out
        else:
            x = conv2d_transpose(x, kernel=5, strides=2, features=filters)
            x = tf.nn.leaky_relu(x)
            x = pixel_norm(x)
            rgb_new = to_rgb(x, channels)
            out = combine(prev_rgb, rgb_new, current_lod, layer_lod, alpha)
            return x, out


def conv2d_transpose(x, kernel, strides, features):
    w = get_weight([kernel, kernel, features, x.shape[-1].value], use_wscale=True)
    w = tf.cast(w, x.dtype)
    os = [tf.shape(x)[0], x.shape[1] * strides, x.shape[2] * strides, features]
    x = tf.nn.conv2d_transpose(x, w, os, strides=[1, strides, strides, 1], padding='SAME')
    return x


def to_rgb(x, channels):
    with tf.variable_scope(f'to_rgb', reuse=tf.AUTO_REUSE):
        x = conv2d_transpose(x, kernel=1, strides=1, features=channels)
        x = tf.nn.leaky_relu(x)
        return x


def combine(old_rgb, new_rgb, current_lod, layer_lod, alpha):
    new_size = tf.shape(new_rgb)[1:3]
    upscale_old_rgb = tf.image.resize_images(old_rgb, new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    combine_old_new = upscale_old_rgb + alpha * (new_rgb - upscale_old_rgb)
    x = tf.cond(current_lod < layer_lod,
        lambda: upscale_old_rgb,
        lambda: tf.cond(tf.equal(layer_lod, current_lod), lambda: combine_old_new, lambda: new_rgb)
    )
    return x



def train(gen, lr, beta1, beta2):
    loss = tf.reduce_mean(-1 * gen)
    optimiser = tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2)\
        .minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator"))
    return loss, optimiser
