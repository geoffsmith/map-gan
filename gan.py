from datetime import datetime
import numpy as np
import tensorflow as tf
from iwgan import generator
from iwgan import discriminator
from iwgan import gradient_penalty
from util.image import save_image
from util.data import get_training_data

tf.logging.set_verbosity(tf.logging.INFO)


def main():
    critic_iterations = 5
    batch_size = 32
    channels = 3
    lr = 0.0001
    beta1 = 0
    beta2 = 0.9
    lmbda = 10
    epochs = 500_000
    log_freq = 1000
    save_image_freq = 10_000
    Z_size = 128

    X_train, iterator = get_training_data(batch_size, channels)
    Z = tf.random_normal(shape=(batch_size, Z_size))

    gen = generator.generator(Z, channels=channels)
    real_dis = discriminator.discriminator(X_train)
    fake_dis = discriminator.discriminator(gen)
    penalty_X = gradient_penalty.gradient_penalty(X_train, gen)
    penalty_dis = discriminator.discriminator(penalty_X)

    d_loss, d_train = discriminator.train(real_dis, fake_dis, penalty_dis, lmbda, penalty_X, lr, beta1, beta2)
    g_loss, g_train = generator.train(fake_dis, lr, beta1, beta2)

    summaries = create_summaries(d_loss, g_loss, gen)

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        writer = create_filewriter(sess)
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(iterator.initializer)

        for epoch in range(epochs):
            for _ in range(critic_iterations):
                d_train_v, d_loss_v = sess.run((d_train, d_loss))
            g_train_v, g_loss_v = sess.run((g_train, g_loss))

            if epoch % log_freq == 0:
                print(f'Epoch: {epoch}\tD loss: {d_loss_v}\tG loss: {g_loss_v}')
                summary = sess.run(summaries)
                writer.add_summary(summary, epoch)

            if epoch % save_image_freq == 0:
                gen_images = sess.run(gen)[:16]
                print(f'Generated images: {gen_images.shape}')
                save_image(epoch, gen_images)


        writer.close()

    return


def create_filewriter(sess):
    path = datetime.now().strftime('%Y%m%d_%H%M')
    return tf.summary.FileWriter(f'logs/{path}/', sess.graph)


def create_summaries(d_loss, g_loss, gen):
    tf.summary.scalar('Discriminator loss', d_loss)
    tf.summary.scalar('Generator loss', g_loss)
    tf.summary.image('Samples', gen[:16], max_outputs=16)
    return tf.summary.merge_all()


if __name__ == '__main__':
    # tf.app.run()
    main()
