import numpy as np
import tensorflow as tf
from mapgan.layers import get_weight, pixel_norm, apply_bias, dense, conv2d

def generator(z, lod, alpha, channels=3):
    dim = 32
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        x = z
        x, out = layer(x, lod, alpha, 0, 3 * dim, channels)      # 8x8
        x, out = layer(x, lod, alpha, 1, 3 * dim, channels, out) # 16 x 16
        x, out = layer(x, lod, alpha, 2, 2 * dim, channels, out, bypass=False) # 32 x 32
        x, out = layer(x, lod, alpha, 3, 1 * dim, channels, out, bypass=False) # 64 x 64
        _, out = layer(x, lod, alpha, 4, 1 * dim, channels, out, bypass=False) # 64 x 64
        out = tf.nn.sigmoid(out)
        tf.summary.histogram('generator out', out)

        return out


def layer(x, current_lod, alpha, layer_lod, filters, channels, prev_rgb=None, bypass=False):
    with tf.variable_scope('lod-' + str(layer_lod), reuse=tf.AUTO_REUSE):
        if layer_lod == 0:
            x = pixel_norm(x, axis=1)
            with tf.variable_scope('dense', reuse=tf.AUTO_REUSE):
                x = dense(x, units=filters * 4 * 4)
                x = tf.reshape(x, shape=(-1, 4, 4, filters))
                x = pixel_norm(tf.nn.leaky_relu(apply_bias(x)))
            with tf.variable_scope('conv', reuse=tf.AUTO_REUSE):
                x = conv2d(x, kernel=3, features=filters)
                x = pixel_norm(tf.nn.leaky_relu(apply_bias(x)))
            out = to_rgb(x, channels)
            return x, out
        else:
            if bypass:
                out = upscale2d(prev_rgb)
                return x, out
            with tf.variable_scope('conv_up', reuse=tf.AUTO_REUSE):
                x = conv2d_transpose(x, kernel=3, strides=2, features=filters)
                x = pixel_norm(tf.nn.leaky_relu(apply_bias(x)))
            with tf.variable_scope('conv', reuse=tf.AUTO_REUSE):
                x = conv2d(x, kernel=3, features=filters)
                x = pixel_norm(tf.nn.leaky_relu(apply_bias(x)))
            rgb_new = to_rgb(x, channels)
            out = combine(prev_rgb, rgb_new, current_lod, layer_lod, alpha)
            return x, out


def conv2d_transpose(x, kernel, strides, features, gain=np.sqrt(2), use_wscale=True):
    w = get_weight([kernel, kernel, features, x.shape[-1].value], use_wscale=use_wscale, fan_in=(kernel**2)*x.shape[1].value, gain=gain)
    w = tf.cast(w, x.dtype)
    # tf.summary.histogram('weight', w)
    os = [tf.shape(x)[0], x.shape[1] * strides, x.shape[2] * strides, features]
    x = tf.nn.conv2d_transpose(x, w, os, strides=[1, strides, strides, 1], padding='SAME')
    return x


def to_rgb(x, channels):
    with tf.variable_scope('to_rgb', reuse=tf.AUTO_REUSE):
        x = conv2d_transpose(x, kernel=1, strides=1, features=channels, gain=1)
        x = apply_bias(x)
        return x


def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
        x = tf.tile(x, [1, 1, factor, 1, factor, 1])
        x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
        return x


def combine(old_rgb, new_rgb, current_lod, layer_lod, alpha):
    upscale_old_rgb = upscale2d(old_rgb)
    d = alpha * (new_rgb - upscale_old_rgb)
    combine_old_new = upscale_old_rgb + d
    # tf.summary.histogram('upscale_old_rgb', upscale_old_rgb)
    # tf.summary.histogram('d', d)
    cond = current_lod < layer_lod
    cond_equal = tf.equal(layer_lod, current_lod)
    x = tf.cond(cond,
        lambda: upscale_old_rgb,
        lambda: tf.cond(cond_equal, lambda: combine_old_new, lambda: new_rgb)
    )
    return x



def train(gen, lr, beta1, beta2):
    loss = tf.reduce_mean(-1 * gen)
    print('gen vars:', tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator"))
    optimiser = tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2)
    minimise = optimiser.minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator"))
    grads = optimiser.compute_gradients(loss)
    # for g in grads:
    #     if g[1].name.startswith('generator'):
    #         print(g)
    # [tf.summary.histogram(f'gradients/{g[1].name}', g[0]) for g in grads if g[1].name.startswith('generator') and g[0] is not None]
    return loss, minimise
