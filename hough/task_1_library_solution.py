import cv2
import numpy as np

import hough.common as common


def read_img(img_path='./res/marker_cut_gray_42.png'):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def main(input_fname, output_fname):

    img = cv2.imread(input_fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 10, 10, apertureSize=3)
    ret, edges = cv2.threshold(edges, 200, 255, cv2.THRESH_BINARY)

    minLineLength = 10
    lines = cv2.HoughLines(edges, 1, np.pi / 100, 200, minLineLength)
    img = common.draw_lines_on_img(lines, img)

    cv2.imwrite(output_fname, img)


if __name__ == '__main__':
    main('../res/marker_cut_rgb_512.png', './output/library_solution.jpg')