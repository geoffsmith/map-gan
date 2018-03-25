import math
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
    critic_iterations = 1
    batch_size = 32
    channels = 3
    lr = 0.001
    beta1 = 0
    beta2 = 0.99
    lmbda = 10
    e_drift = 0.001
    epochs = 200_000
    lod_period = 500_000
    log_freq = 100
    save_image_freq = log_freq * 1
    Z_size = 256

    std_mean = 0.9076
    std_std = 0.0235

    X_train, iterator = get_training_data(batch_size, channels)
    X_train = (X_train - std_mean) / std_std
    Z = tf.random_normal(shape=(batch_size, Z_size))
    lod = tf.placeholder(tf.int32, shape=(), name='lod')
    tf.summary.scalar('lod', lod)
    alpha = tf.placeholder(tf.float32, shape=(), name='alpha')
    tf.summary.scalar('alpha', alpha)

    gen = generator.generator(Z, lod, alpha, channels=channels)
    print('gen shape', gen.shape)
    real_dis = discriminator.discriminator(X_train, lod, alpha)
    fake_dis = discriminator.discriminator(gen, lod, alpha)
    penalty_X = gradient_penalty.gradient_penalty(X_train, gen)
    penalty_dis = discriminator.discriminator(penalty_X, lod, alpha)

    d_loss, d_train = discriminator.train(real_dis, fake_dis, penalty_dis, lmbda, penalty_X, lr, beta1, beta2, e_drift)
    g_loss, g_train = generator.train(fake_dis, lr, beta1, beta2)

    summaries = create_summaries(d_loss, g_loss, gen, std_mean, std_std)

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        writer = create_filewriter(sess)
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(iterator.initializer)

        for epoch in range(epochs):
            lod_val, alpha_val = schedule(epoch, lod_period)
            feed = {lod: lod_val, alpha: alpha_val}
            #print(f'epoch: {epoch}, lod: {lod_val}, alpha: {alpha_val}')
            for _ in range(critic_iterations):
                d_train_v, d_loss_v = sess.run((d_train, d_loss), feed_dict=feed)
            g_train_v, g_loss_v = sess.run((g_train, g_loss), feed_dict=feed)

            if epoch % log_freq == 0 and epoch > 0:
                print(f'Epoch: {epoch}\tD loss: {d_loss_v}\tG loss: {g_loss_v}')
                print(f'epoch: {epoch}, lod: {lod_val}, alpha: {alpha_val}')
                summary = sess.run(summaries, feed_dict=feed)
                writer.add_summary(summary, epoch)

            if epoch % save_image_freq == 0:
                image_out = gen[:16] * std_std + std_mean
                image_out = tf.clip_by_value(image_out, 0.0, 1.0)
                gen_images = sess.run(image_out, feed_dict=feed)[:16]
                save_image(epoch, gen_images)

        writer.close()

    return


def schedule(epoch, lod_period):
    # 500 train, 500 blend
    e = epoch + (lod_period / 2)
    lod = math.floor(e / lod_period)
    alpha = (e % lod_period) / lod_period
    alpha = min(1, alpha * 2)
    return lod, alpha


def create_filewriter(sess):
    path = datetime.now().strftime('%Y%m%d_%H%M')
    return tf.summary.FileWriter(f'logs/{path}/', sess.graph)


def create_summaries(d_loss, g_loss, gen, std_mean, std_std):
    tf.summary.scalar('Discriminator loss', d_loss)
    tf.summary.scalar('Generator loss', g_loss)
    image_out = gen[:12] * std_std + std_mean
    image_out = tf.clip_by_value(image_out, 0.0, 1.0)
    tf.summary.image('Samples', image_out, max_outputs=16)
    return tf.summary.merge_all()


if __name__ == '__main__':
    # tf.app.run()
    main()
