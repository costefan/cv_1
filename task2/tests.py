from utils.gaussian_blur import generate_gaussian_kernel
from utils.convolution import convolution_2d
import cv2
import numpy as np


img = cv2.imread('./../images/marker_cut_rgb_512.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gaussian_kernel = generate_gaussian_kernel(31, 3.0)
res = convolution_2d(gray, gaussian_kernel, 0)
print(res.shape)
