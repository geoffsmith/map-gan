import tensorflow as tf


def gradient_penalty(X, gen):
    print(X.get_shape())
    epsilon = tf.random_uniform(shape=(X.get_shape().as_list()[0],))
    epsilon = tf.expand_dims(epsilon, axis=-1)
    epsilon = tf.expand_dims(epsilon, axis=-1)
    epsilon = tf.expand_dims(epsilon, axis=-1)
    x = X * epsilon + gen * (1.0 - epsilon)
    return x
