import numpy as np


def convolution_2d(image, kernel):
    m, n = kernel.shape
    if m != n:
        raise Exception('Kernel is not square')
    h, w = image.shape
    if w != h:
        raise Exception('Image is not square')

    # create padding
    pad_size = (((h + m * 2) // m * m) - h) // 2
    image = np.pad(image, (pad_size, pad_size), 'constant', constant_values=0)

    # convolve
    new_image = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            k = np.multiply(kernel, image[i:i+n, j:j+m])
            new_image[i, j] = k.sum(axis=(0, 1))

    return new_image
