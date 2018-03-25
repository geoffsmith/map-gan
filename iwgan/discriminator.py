import numpy as np
import tensorflow as tf
from .layers import get_weight, dense, conv2d, apply_bias

def discriminator(X, current_lod, alpha, dim=64, kernel_size=5):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        x = X
        tf.summary.histogram(f'x', x)
        print('x', x.shape)
        x, downsample = layer(None, x, 4, current_lod, alpha, dim, dim) # 32 x 32
        print('x', x.shape)
        x, downsample = layer(x, downsample, 3, current_lod, alpha, dim, 2 * dim) # 16 x 16
        print('x', x.shape)
        x, downsample = layer(x, downsample, 2, current_lod, alpha, 2 * dim, 4 * dim) # 8 x 8
        print('x', x.shape)
        x, downsample = layer(x, downsample, 1, current_lod, alpha, 4 * dim, 8 * dim) # 4 x 4
        print('x', x.shape)
        x, _          = layer(x, downsample, 0, current_lod, alpha, 8 * dim, 8 * dim) # 4 x 4
        print('x', x.shape)

        tf.summary.histogram(f'result', x)
        return x


def layer(x, downsample, lod, current_lod, alpha, features_in, features_out):
    with tf.variable_scope(f'layer-{lod}', reuse=tf.AUTO_REUSE):
        with tf.variable_scope(f'from_rgb', reuse=tf.AUTO_REUSE):
            from_rgb = tf.nn.leaky_relu(apply_bias(conv2d(downsample, 1, features_in)))

        if x is not None:
            combined = from_rgb + alpha * (x - from_rgb)
            x = tf.cond(
                tf.greater_equal(lod, current_lod),
                lambda: from_rgb,
                lambda: tf.cond(
                    tf.equal(lod, current_lod - 1),
                    lambda: combined,
                    lambda: x
                )
            )
        else:
            x = from_rgb

        with tf.variable_scope(f'Conv', reuse=tf.AUTO_REUSE):
            if lod == 0:
                with tf.variable_scope(f'conv', reuse=tf.AUTO_REUSE):
                    x = tf.nn.leaky_relu(apply_bias(conv2d(x, 3, features_out)))
                with tf.variable_scope(f'dense0', reuse=tf.AUTO_REUSE):
                    x = tf.reshape(x, shape=(-1, features_out * 4 * 4))
                    x = tf.nn.leaky_relu(apply_bias(dense(x, features_out)))
                with tf.variable_scope(f'dense1', reuse=tf.AUTO_REUSE):
                    x = apply_bias(dense(x, units=1))
            else:
                with tf.variable_scope(f'conv', reuse=tf.AUTO_REUSE):
                    x = tf.nn.leaky_relu(apply_bias(conv2d(x, 3, features_out)))
                with tf.variable_scope(f'conv_down', reuse=tf.AUTO_REUSE):
                    x = tf.nn.leaky_relu(apply_bias(conv2d(x, 3, features_out)))
                    x = tf.layers.average_pooling2d(x, pool_size=2, strides=2)

        with tf.variable_scope(f'Downsample', reuse=tf.AUTO_REUSE):
            downsample = tf.layers.average_pooling2d(downsample, pool_size=2, strides=2)
        return x, downsample


def train(real_dis, fake_dis, penalty_dis, lmda, penalty_X, lr, beta1, beta2, e_drift):
    penalty_gradients = tf.gradients(penalty_dis, penalty_X)
    norm = tf.norm(penalty_gradients[0], axis=(1, 2))
    loss = tf.reduce_mean(fake_dis - real_dis + lmda * tf.pow(norm - 1, 2) + e_drift * tf.square(real_dis))
    print('disc vars:', tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator"))
    optimiser = tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2)
    minimiser = optimiser.minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator"))
    grads = optimiser.compute_gradients(loss)
    # for g in grads:
    #     if g[1].name.startswith('generator'):
    #         print(g)
    [tf.summary.histogram(f'disc-gradients/{g[1].name}', g[0]) for g in grads if g[1].name.startswith('discriminator') and g[0] is not None]
    return loss, minimiser


