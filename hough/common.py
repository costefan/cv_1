import numpy as np
import cv2


def draw_lines_on_img(lines, img):
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            print((x1, y1), (x2, y2))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img
