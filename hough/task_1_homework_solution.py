import cv2
import os
import numpy as np
from scipy.signal import convolve2d

from utils.gaussian_blur import gaussian_blur

import hough.common as common

DEBUG = True

# Tunable constants
DEG_STEP = 1
N_LARGEST = 10
EDGE_THRESHOLD = 100


IMG_DIR_PATH = os.path.join(os.path.dirname(__file__), 'output')


def k_largest_index_argsort(a, k):
    """Finding largest values indexes in 2d numpy array
    :param a: 
    :param k: 
    :return: 
    """
    idx = np.argsort(a.ravel())[:-k-1:-1]
    return np.column_stack(np.unravel_index(idx, a.shape))


def read_img(img_path='./res/marker_cut_gray_42.png'):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def show_img(image):
    cv2.imshow('img', image)
    cv2.waitKey()


def edge_detection(img):
    """
    Edge detection using Sobel
    operator
    :param img: 
    :return: 
    """
    G_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ])
    G_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ])
    # if it was required not to use convolve2d,
    #  task2 has convolution_2d implemented
    img_g_x = convolve2d(img, G_x)
    img_g_y = convolve2d(img, G_y)

    return np.sqrt(img_g_x ** 2 + img_g_y ** 2)


def hough_transform(img, deg_step=DEG_STEP):
    """Hough transform
    :param img: 
    :param deg_step: 
    :return: 
    """
    # brootforce vals
    thetas = np.deg2rad(np.arange(-90.0, 90.0, deg_step))
    r_length = int(round(np.sqrt(img.shape[0] ** 2 +
                                 img.shape[1] ** 2)))
    rhos = np.arange(-r_length, r_length, 1)

    thetas_count = len(thetas)
    hough_acc = np.zeros((2 * r_length, thetas_count))

    for y, x in zip(*np.nonzero(img)):
        for t_ix, theta in enumerate(thetas):
            rho = int(round(x * np.cos(theta) + y * np.sin(theta))) + r_length
            hough_acc[rho][t_ix] += 1

    return hough_acc, thetas, rhos


def main(input_fname, output_fname):
    """
    Main function
    Steps:
    read_image
    gaussian blur
    edge_detection
    threshold on edges image
    hough transform
    hough space points detection
    output results
    
    :param input_fname: 
    :param output_fname: 
    :return: 
    """
    img = read_img(input_fname)
    img = gaussian_blur(img, (3, 3), 0.9)

    cv2.imwrite(IMG_DIR_PATH + '/after_blur.png', img)

    edges = edge_detection(img).astype('uint8')

    cv2.imwrite(IMG_DIR_PATH + '/after_edge_det.png', edges)

    ret, edges = cv2.threshold(edges, EDGE_THRESHOLD, 255, cv2.THRESH_BINARY)

    cv2.imwrite(IMG_DIR_PATH + '/after_threshold.png', edges)

    hough_acc, thetas, rhos = hough_transform(edges)
    # founding n maximum
    if DEBUG:
        print(k_largest_index_argsort(hough_acc, N_LARGEST))
        print(hough_acc.shape)

    rhos_max, thetas_max = [], []
    for rh_ix, t_ix in k_largest_index_argsort(hough_acc, N_LARGEST):
        rhos_max.append(rhos[rh_ix])
        thetas_max.append(thetas[t_ix])

    img = common.draw_lines_on_img([zip(rhos_max, thetas_max)], img)

    cv2.imwrite(output_fname, img)


