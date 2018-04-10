import tensorflow as tf

def get_training_data(train_data_path, batch_size, channels):
    # path = './data/train.tfrecords'
    # path = 'gs://map-gan/train.tfrecords'
    x = tf.data.TFRecordDataset(train_data_path)
    x = x.map(lambda e: decode(e, channels))
    iterator = x\
        .shuffle(buffer_size=10000)\
        .apply(tf.contrib.data.batch_and_drop_remainder(batch_size))\
        .repeat()\
        .make_initializable_iterator()
    X_train = iterator.get_next()
    return X_train, iterator


def decode(example, channels):
    features = tf.parse_single_example(
        example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(features['image_raw'], tf.float64)
    image = tf.reshape(image, [64, 64, channels])
    image = tf.cast(image, tf.float32)
    image = image * 2.0 - 1.0
    return image
