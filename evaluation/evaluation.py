from math import log10
from math import sqrt

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def PSNR(original, compressed):
    original = original.astype(np.float32)
    compressed = compressed.astype(np.float32)
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def SSIM(original, compressed):
    grayA = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)

    ssim = structural_similarity(grayA, grayB, data_range=grayB.max() - grayB.min())
    return ssim
