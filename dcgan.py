from datetime import datetime
import numpy as np
import tensorflow as tf
from dcgan import generator
from dcgan import discriminator
from util.image import save_image

tf.logging.set_verbosity(tf.logging.INFO)


def main():
    batch_size = 32
    channels = 1
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    iterator = dataset.shuffle(buffer_size=10000).apply(tf.contrib.data.batch_and_drop_remainder(batch_size)).repeat().make_initializable_iterator()
    X_train, _ = iterator.get_next()
    X_train = tf.reshape(X_train, shape=(-1, 28, 28, channels))

    Z = tf.random_normal(shape=(batch_size, 128))
    gen = generator.generator(Z, channels=channels)
    real_dis = discriminator.discriminator(X_train)
    fake_dis = discriminator.discriminator(gen)
    d_loss, d_train = discriminator.train(real_dis, fake_dis)
    g_loss, g_train = generator.train(fake_dis)
    tf.summary.scalar('Discriminator loss', d_loss)
    tf.summary.scalar('Generator loss', g_loss)
    tf.summary.image('Samples', gen[:16], max_outputs=16)
    summaries = tf.summary.merge_all()
    print([x.name for x in tf.global_variables()])

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        writer = create_filewriter(sess)
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(iterator.initializer)

        for epoch in range(10_000):
            d_train_v, d_loss_v = sess.run((d_train, d_loss))
            g_train_v, g_loss_v = sess.run((g_train, g_loss))

            if epoch % 10 == 0:
                print(f'Epoch: {epoch}\tD loss: {d_loss_v}\tG loss: {g_loss_v}')
                summary = sess.run(summaries)
                writer.add_summary(summary, epoch)

            if epoch % 1000 == 0:
                gen_images = sess.run(gen)[:16]
                print(f'Generated images: {gen_images.shape}')
                save_image(epoch, gen_images)


        writer.close()

    return


def create_filewriter(sess):
    path = datetime.now().strftime('%Y%m%d_%H%M')
    return tf.summary.FileWriter(f'logs/{path}/', sess.graph)


if __name__ == '__main__':
    # tf.app.run()
    main()