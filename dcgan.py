import numpy as np
import tensorflow as tf
from dcgan import generator
from dcgan import discriminator

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(X_train, Y_train, mode):
    input_layer = tf.reshape(X_train, [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == 'train')
    logits = tf.layers.dense(inputs=dropout, units=10)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=Y_train, logits=logits)
    optimiser = tf.train.GradientDescentOptimizer(0.001)
    train = optimiser.minimize(loss)
    return train, loss

    # predictions = {
    #     'classes': tf.argmax(input=logits, axis=1),
    #     'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    # }

    # if mode == tf.estimator.ModeKeys.PREDICT:
    #     return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # if mode == tf.estimator.ModeKeys.TRAIN:
    #     optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    #     train_op = optimiser.minimize(
    #         loss=loss,
    #         global_step=tf.train.get_global_step()
    #     )
    #     return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # eval_metric_ops = {
    #     'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
    # }

    # return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    batch_size = 32
    channels = 1
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # eval_data = mnist.test.images
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    iterator = dataset.shuffle(buffer_size=10000).batch(batch_size).repeat().make_initializable_iterator()
    X_train, Y_train = iterator.get_next()
    X_train = tf.reshape(X_train, shape=(-1, 28, 28, channels))

    print('X_Train', X_train.shape)

    Z = tf.random_normal(shape=(batch_size, 128))
    # Z = tf.placeholder(tf.float32, shape=(None, 128))
    # X = tf.placeholde(tf.float32, shape=(None, 32, 32, channels))
    # with tf.device('/gpu:0'):
    gen = generator.generator(Z, channels=channels)
    print('gen shape', gen)
    real_dis = discriminator.discriminator(X_train)
    fake_dis = discriminator.discriminator(gen)
    d_loss, d_train = discriminator.train(real_dis, fake_dis)
    g_loss, g_train = generator.train(gen)
    print([x.name for x in tf.global_variables()])

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        writer = tf.summary.FileWriter('logs', sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(iterator.initializer)

        for epoch in range(10_000):
            # z = np.random.normal(size=(batch_size, 128))
            d_train_v, d_loss_v = sess.run((d_train, d_loss))
            g_train_v, g_loss_v = sess.run((g_train, g_loss))
            if epoch % 100 == 0:
                print(f'Epoch: {epoch}\tD Train: {d_train_v}, loss: {d_loss_v}\tG loss: {g_loss_v}')

        writer.close()

    return

    train, loss = cnn_model_fn(X_train, Y_train, 'train')

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(iterator.initializer)

        for i in range(20000):
            _, loss_value = sess.run((train, loss))
            if i % 100 == 0:
                print(f'Epoch {i}: loss {loss_value}')


    # mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir='./model/')

    # tensors_to_log = { 'probabilities': 'softmax_tensor' }
    # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={'x': train_data},
    #     y=train_labels,
    #     batch_size=100,
    #     num_epochs=None,
    #     shuffle=True
    # )
    # mnist_classifier.train(input_fn=train_input_fn, steps=20000)

    # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={'x': eval_data},
    #     y=eval_labels,
    #     num_epochs=1,
    #     shuffle=False
    # )
    # eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    # print(eval_results)


if __name__ == '__main__':
    # tf.app.run()
    main()