import math
import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf

from mapgan import generator
from mapgan import discriminator
from mapgan import gradient_penalty
from mapgan.data import get_training_data
from mapgan.image import save_image
from mapgan.tiled_images import build_visualization

tf.logging.set_verbosity(tf.logging.INFO)

def main(args):
    critic_iterations = 1
    batch_size = 32
    channels = 1
    lr = 0.001
    beta1 = 0
    beta2 = 0.99
    lmbda = 10
    e_drift = 0.001
    epochs = args.epochs
    lod_period = 80000
    log_freq = 1000
    save_image_freq = args.save_image_freq
    Z_size = 256
    job_dir = args.job_dir
    train_data_path = args.train_data

    X_train, iterator = get_training_data(train_data_path, batch_size, channels)
    Z = tf.random_normal(shape=(batch_size, Z_size))
    lod = tf.placeholder(tf.int32, shape=(), name='lod')
    alpha = tf.placeholder(tf.float32, shape=(), name='alpha')
    tf.summary.scalar('2_blending/lod', lod)
    tf.summary.scalar('2_blending/alpha', alpha)

    gen = generator.generator(Z, lod, alpha, channels=channels)
    real_dis = discriminator.discriminator(X_train, lod, alpha)
    fake_dis = discriminator.discriminator(gen, lod, alpha)
    penalty_X = gradient_penalty.gradient_penalty(X_train, gen)
    penalty_dis = discriminator.discriminator(penalty_X, lod, alpha)

    d_loss, d_train = discriminator.train(real_dis, fake_dis, penalty_dis, lmbda, penalty_X, lr, beta1, beta2, e_drift)
    g_loss, g_train = generator.train(fake_dis, lr, beta1, beta2)

    summaries = create_summaries(d_loss, g_loss, gen)

    saver = tf.train.Saver()

    config = tf.ConfigProto(log_device_placement=False)
    with tf.Session(config=config) as sess:
        writer = create_filewriter(sess, job_dir)
        if args.restore is None:
            init = tf.global_variables_initializer()
            sess.run(init)
            first_epoch = 0
        else:
            checkpoint = tf.train.latest_checkpoint(checkpoint_path(job_dir))
            print('checkpoint', checkpoint)
            saver.restore(sess, checkpoint)
            first_epoch = args.restore

        sess.run(iterator.initializer)

        for epoch in range(first_epoch, epochs):
            lod_val, alpha_val = schedule(epoch, lod_period)
            feed = {lod: lod_val, alpha: alpha_val}
            for _ in range(critic_iterations):
                d_train_v, d_loss_v = sess.run((d_train, d_loss), feed_dict=feed)
            g_train_v, g_loss_v = sess.run((g_train, g_loss), feed_dict=feed)

            if epoch % log_freq == 0 and epoch > 0:
                print('Epoch:', epoch, '\tD loss: ', d_loss_v, '\tG loss:', g_loss_v)
                saver.save(sess, checkpoint_prefix(job_dir), global_step=epoch)
                summary = sess.run(summaries, feed_dict=feed)
                writer.add_summary(summary, epoch)

            if save_image_freq is not None and epoch % save_image_freq == 0:
                gen_images = sess.run(gen, feed_dict=feed)[:16]
                save_image(epoch, gen_images)

        writer.close()

    return


def checkpoint_path(job_dir):
    return job_dir + 'checkpoints/'


def checkpoint_prefix(job_dir):
    return checkpoint_path(job_dir) + 'model.ckpt'



def schedule(epoch, lod_period):
    e = epoch + (lod_period / 2)
    lod = math.floor(e / lod_period)
    alpha = (e % lod_period) / lod_period
    alpha = min(1, alpha * 2)
    return lod, alpha


def create_filewriter(sess, job_dir):
    path = datetime.now().strftime('%Y%m%d_%H%M')
    path = job_dir + '/logs/' + path + '/'
    print('log path:', path)
    return tf.summary.FileWriter(path, sess.graph)


def create_summaries(d_loss, g_loss, gen):
    tf.summary.scalar('0_losses/d_loss', d_loss)
    tf.summary.scalar('0_losses/g_loss', g_loss)
    tiled = (gen[:16] + 1.0) * (255.0 / 2.0)
    tiled = tf.cast(tiled, tf.uint8)
    tiled = build_visualization(tiled, 4)
    print('tiled shape', tiled.shape)
    tf.summary.image('samples', tiled)
    return tf.summary.merge_all()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir')
    parser.add_argument('--train-data')
    parser.add_argument('--epochs', default=300000, type=int)
    parser.add_argument('--save-image-freq', default=None, type=int)
    parser.add_argument('--restore', default=None, type=int)
    args = parser.parse_args()
    main(args)
