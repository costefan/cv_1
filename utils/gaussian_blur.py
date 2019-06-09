import numpy as np

from utils.convolution import convolution_2d
from scipy.signal import convolve2d

def gaussian(x, mu, sigma):
    return np.exp(-(((x - mu) / (sigma)) ** 2) / 2.0)


def generate_gaussian_kernel(kernel_size, sigma):
    kernel_h = [gaussian(x, (kernel_size - 1) / 2, sigma)
                for x in range(kernel_size)]
    kernel_v = [x for x in kernel_h]
    kernel_2d = [[xh * xv for xh in kernel_h] for xv in kernel_v]
    kernel_2d_norm = kernel_2d / np.sum(kernel_2d)

    return kernel_2d_norm


def gaussian_blur(img, kernel_sz, sigma):
    if not kernel_sz[0] == kernel_sz[1]:
        raise Exception('Kernel is not square')

    gaussian_kernel = generate_gaussian_kernel(kernel_sz[0], sigma)

    res = convolve2d(img, gaussian_kernel)

    return res
