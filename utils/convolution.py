import numpy as np


def convolution_2d(image, kernel):
    m, n = kernel.shape
    if m != n:
        raise Exception('Kernel is not square')
    h, w = image.shape
    if w != h:
        raise Exception('Image is not square')
    new_h = h - m + 1
    new_w = w - m + 1
    new_image = np.zeros((new_h, new_w))
    for i in range(h):
        for j in range(w):
            new_image[i][j] = np.sum(image[i:i+m, j:j+m] * kernel)

    return new_image
