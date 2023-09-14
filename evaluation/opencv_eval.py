import numpy as np
import cv2

from evaluation.evaluation import PSNR, SSIM
from utils.image_utils import center_crop


def evaluate_psnr(dataset_path, gt, scale_factor, center_crop_dim, translations, interpolation):
    best_psnr = -np.inf
    best_img = None
    avg_opencv_psnr = 0.0
    min_psnr = np.inf

    for i in range(len(translations)):
        img = cv2.imread(f"{dataset_path}/lrs/lr{i}.png")
        lr_h, lr_w = img.shape[:2]
        hr_h, hr_w = int(lr_h * scale_factor), int(lr_w * scale_factor)

        img = cv2.resize(img, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
        img = cv2.warpAffine(img, translations[i], (hr_w, hr_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        cropped = center_crop(img, center_crop_dim)
        
        psnr = PSNR(gt, cropped)
        avg_opencv_psnr += psnr
        if best_psnr < psnr:
            best_img = cropped
            best_psnr = psnr
        if min_psnr > psnr:
            min_psnr = psnr
    avg_opencv_psnr /= len(translations)
    return best_psnr, best_img, avg_opencv_psnr, min_psnr



def evaluate_ssim(dataset_path, gt, scale_factor, center_crop_dim, translations, interpolation):
    best_ssim = -np.inf
    best_img = None
    avg_opencv_ssim = 0.0
    min_ssim = np.inf
    
    for i in range(len(translations)):
        img = cv2.imread(f"{dataset_path}/lrs/lr{i}.png")
        lr_h, lr_w = img.shape[:2]
        hr_h, hr_w = int(lr_h * scale_factor), int(lr_w * scale_factor)
        
        img = cv2.resize(img, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
        img = cv2.warpAffine(img, translations[i], (hr_w, hr_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        cropped = center_crop(img, center_crop_dim)
    
        ssim = SSIM(gt, cropped)
        avg_opencv_ssim += ssim
        if best_ssim < ssim:
            best_img = cropped
            best_ssim = ssim
        
        if min_ssim > ssim:
            min_ssim = ssim

    avg_opencv_ssim /= len(translations)
    return best_ssim, best_img, avg_opencv_ssim, min_ssim


