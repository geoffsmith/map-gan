"""create_train_data.py

Create the training data from input tiff maps

Usage:
    create_train_data.py <input_dir> <output_dir>
"""
from docopt import docopt
from glob import glob
from skimage.io import imsave, imread
from skimage.transform import resize
from skimage import img_as_float
from itertools import islice, chain
import numpy as np
import tensorflow as tf

SUBIMAGE_SIZE = 150
TRAIN_IMG_SIZE = 64


def main():
    arguments = docopt(__doc__)
    create_train_data(arguments['<input_dir>'], arguments['<output_dir>'])


def create_train_data(input_dir, output_dir):
    files = glob(f'{input_dir}/*.tif')
    examples = chain(*(split_image(file) for file in files))
    create_tfrecords(examples, output_dir)
    # i = 0
    # for example_chunk in grouper(5000, examples):
    #     create_train_data_out(example_chunk, output_dir, i)
    #     i += 1


def split_image(file):
    img = img_as_float(imread(file))
    for i in range(0, 5000, SUBIMAGE_SIZE):
        for j in range(0, 5000, SUBIMAGE_SIZE):
            sub_img = img[i:i + SUBIMAGE_SIZE, j:j + SUBIMAGE_SIZE].copy()
            yield resize(sub_img, (TRAIN_IMG_SIZE, TRAIN_IMG_SIZE), mode='constant', preserve_range=True)


def create_train_data_out(examples, dir, i):
    j = 0
    for example in examples:
        imsave(f'{dir}/{i}-{j}.png', example)
        j += 1
    # data = np.vstack([np.resize(example, (1, TRAIN_IMG_SIZE, TRAIN_IMG_SIZE, 3)) for example in examples])
    # filename = f'{dir}/train_{i:03}.npy'
    # print('Saving:', filename)
    # np.save(filename, data)
    

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tfrecords(examples, dir):
    with tf.python_io.TFRecordWriter(f'{dir}/train.tfrecords') as writer:
        for example in examples:
            print(np.min(example), np.max(example))
            height = example.shape[0]
            width = example.shape[1]
            raw = example.tostring()
            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'image_raw': _bytes_feature(raw)
            }))
            writer.write(tf_example.SerializeToString())



def grouper(n, iterable):
    it = iter(iterable)
    while True:
       chunk = tuple(islice(it, n))
       if not chunk:
           return
       yield chunk


if __name__ == '__main__':
    main()
