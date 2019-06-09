import cv2
import numpy as np
from scipy.signal import convolve2d

import hough.common as common

DEBUG = True

# Constants
DEG_STEP = 1
N_LARGEST = 7
EDGE_THRESHOLD = 200


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
    1. read_image
    2. edge_detection
    3. threshold on edges image
    4. hough transform
    5. hough space points detection
    6. output results
    
    :param input_fname: 
    :param output_fname: 
    :return: 
    """
    img = read_img(input_fname)

    edges = edge_detection(img).astype('uint8')
    ret, edges = cv2.threshold(edges, EDGE_THRESHOLD, 255, cv2.THRESH_BINARY)

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

