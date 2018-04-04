"""create_train_data.py

Create the training data from input tiff maps

Usage:
    create_train_data.py [options] <input_dir> <output_dir>

Options:
    --smooth        Apply gaussian smoothing
    --greyscale     Convert images to greyscale
    --debug         Write training examples to pngs and don't write tensorflow records
    -n=<count>      Maximum number of examples
"""
from docopt import docopt
from glob import glob
from skimage.io import imsave, imread
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage import img_as_float
from itertools import islice, chain
import numpy as np
import tensorflow as tf

SUBIMAGE_SIZE = 150
TRAIN_IMG_SIZE = 64


def main():
    arguments = docopt(__doc__)
    create_train_data(arguments['<input_dir>'], arguments['<output_dir>'], arguments)


def create_train_data(input_dir, output_dir, arguments):
    files = glob(f'{input_dir}/*.tif')
    examples = chain(*(split_image(file) for file in files))
    # examples = calculate_moments(examples)
    if arguments['--greyscale']:
        examples = convert_greyscale(examples)
    if arguments['--smooth']:
        examples = apply_smoothing(examples)
    if arguments['-n']:
        examples = islice(examples, int(arguments['-n']))
    if arguments['--debug']:
        create_debug(examples, output_dir)
    else:
        create_tfrecords(examples, output_dir)


def create_debug(examples, output_dir):
    i = 0
    for example in examples:
        imsave(f'{output_dir}/{i}.png', example)
        i += 1


def split_image(file):
    img = img_as_float(imread(file))
    for i in range(0, 5000, SUBIMAGE_SIZE):
        for j in range(0, 5000, SUBIMAGE_SIZE):
            sub_img = img[i:i + SUBIMAGE_SIZE, j:j + SUBIMAGE_SIZE].copy()
            yield resize(sub_img, (TRAIN_IMG_SIZE, TRAIN_IMG_SIZE), mode='constant', preserve_range=True)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tfrecords(examples, dir):
    with tf.python_io.TFRecordWriter(f'{dir}/train.tfrecords') as writer:
        for example in examples:
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


def calculate_moments(examples):
    mean = 0
    std = 0
    k = 0
    for example in examples:
        for x in example.flatten():
            k += 1 
            old_mean = mean
            mean = mean + (x - mean) / k
            std = std + (x - mean) * (x - old_mean)
        # yield example
    print('mean', mean)
    print('std', std / (k - 1))


def convert_greyscale(examples):
    for example in examples:
        yield rgb2gray(example)


def apply_smoothing(examples):
    for example in examples:
        yield gaussian(example)



if __name__ == '__main__':
    main()
