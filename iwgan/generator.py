import numpy as np
import tensorflow as tf

def generator(z, lod, alpha, channels=3):
    dim = 32
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        x = tf.reshape(z, shape=(-1, 1, 1, 8 * dim))
        x, out = layer(x, lod, alpha, 0, 8 * dim, channels)      # 8x8
        x, out = layer(x, lod, alpha, 1, 4 * dim, channels, out) # 16 x 16
        x, out = layer(x, lod, alpha, 2, 2 * dim, channels, out) # 32 x 32
        x, out = layer(x, lod, alpha, 3, 1 * dim, channels, out) # 64 x 64
        _, out = layer(x, lod, alpha, 4, 1 * dim, channels, out) # 64 x 64
        out = tf.nn.sigmoid(out)

        return out


def layer(x, current_lod, alpha, layer_lod, filters, channels, prev_rgb=None):
    if layer_lod == 0:
        x = tf.layers.conv2d_transpose(inputs=x, filters=filters, kernel_size=4, strides=4, padding='same')
        x = tf.nn.leaky_relu(x)
        # x = conv_transpose(x, filters)
        out = to_rgb(x, channels)
        return x, out
    else:
        x = conv_transpose(x, filters)
        rgb_new = to_rgb(x, channels)
        out = combine(prev_rgb, rgb_new, current_lod, layer_lod, alpha)
        return x, out


def conv_transpose(x, filters):
    x = tf.layers.conv2d_transpose(inputs=x, filters=filters, kernel_size=5, strides=2, padding='same')
    x = tf.nn.leaky_relu(x)
    return x


def to_rgb(x, channels):
    return tf.layers.conv2d_transpose(inputs=x, filters=channels, kernel_size=1, padding='same')


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
