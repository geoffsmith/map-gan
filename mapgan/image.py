from skimage.util.montage import montage2d
from skimage.io import imsave
from mapgan.montage import montage
import numpy as np


def save_image(epoch, images):
    print(np.min(images), np.max(images))
    #print(images)
    images = images * 2.0 - 1.0
    images = np.squeeze(images)
    image = montage(images, multichannel=True, padding_width=5)
    imsave('progress/' + epoch + '.jpg', image)
