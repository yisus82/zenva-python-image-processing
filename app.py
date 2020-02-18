# -*- coding: utf-8 -*-
import cv2
import numpy as np
import tkinter
from tkinter.filedialog import askopenfilename, asksaveasfilename

def load_image():
    original_filename = askopenfilename()
    while not original_filename:
        original_filename = askopenfilename()
    
    # read in an image, make a grayscale copy
    color_original = cv2.imread(original_filename)
    gray_original = cv2.cvtColor(color_original, cv2.COLOR_BGR2GRAY)
    
    # create the UI (window and trackbars)
    cv2.namedWindow(original_filename)
    cv2.createTrackbar('Contrast', original_filename, 1, 100, dummy)
    cv2.createTrackbar('Brightness', original_filename, 50, 100, dummy)
    cv2.createTrackbar('Filter', original_filename, 0, len(kernels)-1, dummy)
    cv2.createTrackbar('Grayscale', original_filename, 0, 1, dummy)
    return original_filename, color_original, gray_original

# dummy function that does nothing
def dummy(value):
    pass

# define convolution kernels
identity_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
gaussian_kernel1 = cv2.getGaussianKernel(3, 0)
gaussian_kernel2 = cv2.getGaussianKernel(5, 0)
box_kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.float32) / 9.0

kernels = [identity_kernel, sharpen_kernel, gaussian_kernel1, gaussian_kernel2, box_kernel]

original_filename, color_original, gray_original = load_image()

# main UI loop
while True:
    # read all of the trackbar values
    grayscale = cv2.getTrackbarPos('Grayscale', original_filename)
    contrast = cv2.getTrackbarPos('Contrast', original_filename)
    brightness = cv2.getTrackbarPos('Brightness', original_filename)
    kernel_idx = cv2.getTrackbarPos('Filter', original_filename)

    # apply the filters
    color_modified = cv2.filter2D(color_original, -1, kernels[kernel_idx])
    gray_modified = cv2.filter2D(gray_original, -1, kernels[kernel_idx])
    
    # apply the brightness and contrast
    color_modified = cv2.addWeighted(color_modified, contrast, np.zeros_like(color_original), 0, brightness - 50)
    gray_modified = cv2.addWeighted(gray_modified, contrast, np.zeros_like(gray_original), 0, brightness - 50)
    
    # wait for keypress (100 milliseconds)
    key = cv2.waitKey(100)
    if key == ord('q'):
        break
    elif key == ord('s'):
        copy_filename = asksaveasfilename()
        if not copy_filename is None:
            continue
        # save image
        if grayscale == 0:
            cv2.imwrite(copy_filename, color_modified)
        else:
            cv2.imwrite(copy_filename, gray_modified)
    elif key == ord('o'):
        cv2.destroyWindow(original_filename)
        original_filename, color_original, gray_original = load_image()
    
    # show the image
    if grayscale == 0:
        cv2.imshow(original_filename, color_modified)
    else:
        cv2.imshow(original_filename, gray_modified)

# window cleanup
cv2.destroyAllWindows()
