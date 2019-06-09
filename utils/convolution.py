import numpy as np


def convolution_2d(image, kernel):
    m, n = kernel.shape
    y, x = image.shape
    if m != n:
        raise Exception('Kernel is not square')
    y = y - m + 1
    x = x - m + 1
    new_image = np.zeros((y, x))
    for i in range(y):
        for j in range(x):
            new_image[i][j] = np.sum(image[i:i+m, j:j+m]*kernel)

    return new_image