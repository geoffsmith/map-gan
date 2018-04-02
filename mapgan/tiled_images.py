import tensorflow as tf

def scale_image_colors(samples, N):
    samples = tf.reshape(samples, [N, -1])
    sample_max = tf.reduce_max(samples, axis=1, keep_dims=True)
    sample_min = tf.reduce_min(samples, axis=1, keep_dims=True)
    sample_range = sample_max - sample_min
    samples = (samples - sample_min) / sample_range  # type: tf.Tensor
    samples = tf.image.convert_image_dtype(samples, dtype=tf.uint8, saturate=True)
    samples = tf.reshape(samples, [N, 64, 64, 3])
    return samples

def tile_image_batch(images: tf.Tensor, block: int):
  _, h, w, c = images.get_shape().as_list()
#   assert n == block ** 2
  images = tf.batch_to_space(images, [[0, 0], [0, 0]], block)
  # at this point, we have (in the first column) all the first pixels of all images,
  # then all the second pixels etc, so we need to un-shuffle the images.
  images = tf.reshape(images, [h, block, w, block, c])
  images = tf.transpose(images, [1, 0, 3, 2, 4])
  images = tf.reshape(images, [1, h * block, w * block, c])
  return images
  
def build_visualization(samples: tf.Tensor, visualization_block, png_compression=9):
    # assume that uint8 content is already scaled
    N = visualization_block ** 2
    # if samples.dtype != tf.uint8:
    #   samples = scale_image_colors(samples, N)
    return tile_image_batch(samples, block=visualization_block)
    # visualization_samples = tf.image.encode_png(samples, compression=png_compression)
    # return visualization_samples
