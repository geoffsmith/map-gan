import math
import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
from mapgan import generator
from mapgan import discriminator
from mapgan import gradient_penalty
# from mapgan.image import save_image
from mapgan.data import get_training_data

tf.logging.set_verbosity(tf.logging.INFO)


def main(args):
    critic_iterations = 1
    batch_size = 32
    channels = 3
    lr = 0.0001
    beta1 = 0
    beta2 = 0.99
    lmbda = 10
    e_drift = 0.001
    epochs = 300000
    lod_period = 50000
    log_freq = 1000
    save_image_freq = log_freq * 1
    Z_size = 256
    job_dir = get_job_dir(args.job_dir)
    train_data_path = args.train_data

    std_mean = 0.9076
    std_std = 0.0235

    X_train, iterator = get_training_data(train_data_path, batch_size, channels)
    # X_train = (X_train - std_mean) / std_std
    Z = tf.random_normal(shape=(batch_size, Z_size))
    lod = tf.placeholder(tf.int32, shape=(), name='lod')
    tf.summary.scalar('blending/lod', lod)
    alpha = tf.placeholder(tf.float32, shape=(), name='alpha')
    tf.summary.scalar('blending/alpha', alpha)

    gen = generator.generator(Z, lod, alpha, channels=channels)
    real_dis = discriminator.discriminator(X_train, lod, alpha)
    fake_dis = discriminator.discriminator(gen, lod, alpha)
    penalty_X = gradient_penalty.gradient_penalty(X_train, gen)
    penalty_dis = discriminator.discriminator(penalty_X, lod, alpha)

    d_loss, d_train = discriminator.train(real_dis, fake_dis, penalty_dis, lmbda, penalty_X, lr, beta1, beta2, e_drift)
    g_loss, g_train = generator.train(fake_dis, lr, beta1, beta2)

    summaries = create_summaries(d_loss, g_loss, gen, std_mean, std_std)

    saver = tf.train.Saver()

    config = tf.ConfigProto(log_device_placement=False)
    # config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        writer = create_filewriter(sess, job_dir)
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(iterator.initializer)

        for epoch in range(epochs):
            lod_val, alpha_val = schedule(epoch, lod_period)
            feed = {lod: lod_val, alpha: alpha_val}
            for _ in range(critic_iterations):
                d_train_v, d_loss_v = sess.run((d_train, d_loss), feed_dict=feed)
            g_train_v, g_loss_v = sess.run((g_train, g_loss), feed_dict=feed)

            if epoch % log_freq == 0 and epoch > 0:
                print('Epoch:', epoch, '\tD loss: ', d_loss_v, '\tG loss:', g_loss_v)
                saver.save(sess, job_dir + '/checkpoints/model.ckpt', global_step=epoch)
                summary = sess.run(summaries, feed_dict=feed)
                writer.add_summary(summary, epoch)

            # if epoch % save_image_freq == 0:
            #     # image_out = gen[:16] * std_std + std_mean
            #     # image_out = tf.clip_by_value(image_out, 0.0, 1.0)
            #     gen_images = sess.run(gen, feed_dict=feed)[:16]
            #     save_image(epoch, gen_images)

        writer.close()

    return


def schedule(epoch, lod_period):
    e = epoch + (lod_period / 2)
    lod = math.floor(e / lod_period)
    alpha = (e % lod_period) / lod_period
    alpha = min(1, alpha * 2)
    return lod, alpha


def get_job_dir(arg_job_dir):
    path = arg_job_dir
    if arg_job_dir is None:
        path = datetime.now().strftime('%Y%m%d_%H%M')
        path = 'logs/' + path + '/'
    return path


def create_filewriter(sess, job_dir):
    return tf.summary.FileWriter(job_dir, sess.graph)


def create_summaries(d_loss, g_loss, gen, std_mean, std_std):
    tf.summary.scalar('losses/d_loss', d_loss)
    tf.summary.scalar('losses/g_loss', g_loss)
    # image_out = gen[:12] * std_std + std_mean
    # image_out = tf.clip_by_value(image_out, 0.0, 1.0)
    tf.summary.image('samples', gen[:12], max_outputs=16)
    return tf.summary.merge_all()


if __name__ == '__main__':
    # tf.app.run()
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir')
    parser.add_argument('--train-data')
    args = parser.parse_args()
    main(args)
