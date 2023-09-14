import argparse
import json
import os
import shutil
import time
import random

import cv2
import numpy as np
import torch
from calc_affine_mat.calc_affine_matrix import cal_affine_matrix

from evaluation.opencv_eval import evaluate_psnr, evaluate_ssim
from train import Trainer
from models.encoding import Encoding
from models.network import Network
from utils.model_utils import calculate_xs_and_ys, next_multiple, determine_optimizer, determine_criterion
from utils.image_utils import read_image, center_crop, srgb_to_linear
from evaluation.evaluation import PSNR, SSIM

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def inference(config, image_folder, cropped_sample_coords=[[240, 250], [350, 340]]):
    dataset_path = os.path.join(config["dataset_path"], image_folder)
    scale_factor = config["scale_factor"]
    n_levels = config["n_levels"]
    n_imgs = config["n_imgs"]
    n_features_per_level = config["n_features_per_level"]
    ours_interpolation = config["ours_interpolation"]
    opencv_interpolation = config["opencv_interpolation"]

    print(f"Dataset Path: {dataset_path}")
    print(f"Scale Factor: {scale_factor}")
    print(f"# of Levels: {n_levels}")
    print(f"# of Features: {n_features_per_level}")
    print(f"# of Images: {n_imgs}")
    print(f"Ours Interpolation: {ours_interpolation}")
    print(f"OpenCV Interpolation: {opencv_interpolation}")

    exp_result = os.path.join(config["result_dir"], image_folder)

    t = time.time()
    imgs = []
    translations = []

    base_img = cv2.imread(f"{dataset_path}/lrs/lr0.png")

    lr_h, lr_w = base_img.shape[:2]
    hr_h, hr_w = int(lr_h*scale_factor), int(lr_w*scale_factor)

    if scale_factor == 4:
        T = np.array([[1.0, 0.0, -1.125],
                      [0.0, 1.0, -1.125]]).astype(np.float32)
    
    elif scale_factor == 8:
        T = np.array([[1.0, 0.0, -1.75],
                      [0.0, 1.0, -1.75]]).astype(np.float32)
    else:
        print("No Implementation")
        exit(1)
    
    translations.append(T)

    base_img = cv2.resize(base_img, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
    base_img = cv2.warpAffine(base_img, T, (hr_w, hr_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    base_img_copy = center_crop(base_img, (hr_w-40, hr_h-40))

    base_img_copy = base_img_copy.astype(np.float32)
    base_img_copy /= 255.0
    base_img_copy = cv2.cvtColor(base_img_copy, cv2.COLOR_BGR2RGB)
    if scale_factor == 4:
        imgs.append([srgb_to_linear(base_img_copy), (-1.125/(2*scale_factor), -1.125/(2*scale_factor))])

    elif scale_factor == 8:
        imgs.append([srgb_to_linear(base_img_copy), (-1.75/(2*scale_factor), -1.75/(2*scale_factor))])
    else:
        print("No Implementation")
        exit(1)

    try:
        for i in range(1, n_imgs):
            img = cv2.imread(f"{dataset_path}/lrs/lr{i}.png")
            r_img = cv2.resize(img, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)

            affine_mat = cal_affine_matrix(base_img, r_img)
            
            if scale_factor == 4:
                affine_mat[0][2] -= 1.125 - 1.0
                affine_mat[1][2] -= 1.125 - 1.0
            
            elif scale_factor == 8:
                affine_mat[0][2] -= 1.75 - 1.0
                affine_mat[1][2] -= 1.75 - 1.0
            else:
                print("No Implementation")
                exit(-1)

            translations.append(affine_mat)

            img = cv2.resize(img, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
            img = cv2.warpAffine(img, affine_mat, (hr_w, hr_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            img = center_crop(img, (hr_w-40, hr_h-40))

            img = img.astype(np.float32)
            img /= 255.0
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append([srgb_to_linear(img), (affine_mat[0][2]/(2*scale_factor), affine_mat[1][2]/(2*scale_factor))])

    except Exception as e:
        print(f"\n {str(e)} \n")
        return 0, 0, 0, 0, 0, 0, 0, 0

    print(f"Translation Time: {time.time()-t:.2f} s")

    if os.path.exists(f"{exp_result}/inferences"):
        shutil.rmtree(f"{exp_result}/inferences")
    os.mkdir(f"{exp_result}/inferences")

    height, width = imgs[0][0].shape[:2]
    height = int(height)
    width = int(width)

    n_coords = width * height
    n_coords_padded = next_multiple(n_coords, config["batch_size_granularity"])
    xs_and_ys = calculate_xs_and_ys(width, height, n_coords_padded)

    network = Network(activation_name=config["activation"],
                      n_input=config["n_levels"] * config["n_features_per_level"],
                      n_neurons=config["n_neurons"],
                      n_hidden_layers=config["n_hidden_layers"])

    encoding = Encoding(scale=config["per_level_scale"],
                        n_levels=config["n_levels"],
                        base_resolution=config["base_resolution"],
                        n_features=config["n_features_per_level"],
                        hashmap_size=1 << config["log2_hashmap_size"])

    optimizer = determine_optimizer(config["optimizer"], network.parameters(), config["learning_rate"])
    criterion = determine_criterion(config["loss"])

    trainer = Trainer(encoding=encoding, network=network, optimizer=optimizer, criterion=criterion,
                      n_epochs=config["n_epochs"], xs_and_ys=xs_and_ys, n_coords=n_coords,
                      batch_size=config["batch_size"],
                      scale_factor=scale_factor,
                      width=width, height=height,
                      imgs=imgs,
                      interpolation_type=ours_interpolation,
                      result_dir_path=exp_result)
    t = time.time()      
    trainer.train()
    print(f"Training Time: {time.time()-t:.2f} s")

    gt_img = cv2.imread(f"{dataset_path}/hr.png")
    gt_h, gt_w = gt_img.shape[:2]
    center_crop_dim = (gt_h-40, gt_w-40)

    gt_img = center_crop(gt_img, center_crop_dim)
    
    # gets the latest image
    if not os.path.exists(f"{exp_result}/inferences/inference_best.png"):
        return 0, 0, 0, 0, 0, 0, 0, 0
    final_trained_img = cv2.imread(f"{exp_result}/inferences/inference_best.png", 1)
    final_trained_img = center_crop(final_trained_img, center_crop_dim)

    trained_psnr = PSNR(gt_img, final_trained_img)
    trained_ssim = SSIM(gt_img, final_trained_img)

    # gets the best result
    max_cubic_psnr, best_opencv_img_psnr, avg_cubic_psnr, min_cubic_psnr = evaluate_psnr(dataset_path, gt_img, scale_factor,
                                                                            center_crop_dim, translations,
                                                                            interpolation=opencv_interpolation)
    max_cubic_ssim, best_opencv_img_ssim, avg_cubic_ssim, min_cubic_ssim = evaluate_ssim(dataset_path, gt_img, scale_factor,
                                                                            center_crop_dim, translations,
                                                                            interpolation=opencv_interpolation)
    
    vis = np.concatenate((gt_img, final_trained_img, best_opencv_img_psnr), axis=1)
    cv2.imwrite(f"{exp_result}/combination_of_gt_ours_opencv.png", vis)

    cv2.imwrite(f"{exp_result}/hr.png", gt_img)
    cv2.imwrite(f"{exp_result}/ours.png", final_trained_img)
    cv2.imwrite(f"{exp_result}/opencv.png", best_opencv_img_psnr)

    x1, y1 = cropped_sample_coords[0]
    x2, y2 = cropped_sample_coords[1]

    sample_gt = gt_img[y1:y2, x1:x2]
    sample_gt = cv2.copyMakeBorder(sample_gt, 5, 5, 5, 5, cv2.BORDER_CONSTANT)
    sample_trained = final_trained_img[y1:y2, x1:x2]
    sample_trained = cv2.copyMakeBorder(sample_trained, 5, 5, 5, 5, cv2.BORDER_CONSTANT)
    sample_opencv = best_opencv_img_psnr[y1:y2, x1:x2]
    sample_opencv = cv2.copyMakeBorder(sample_opencv, 5, 5, 5, 5, cv2.BORDER_CONSTANT)
    vis = np.concatenate((sample_gt, sample_trained, sample_opencv), axis=1)
    cv2.imwrite(f"{exp_result}/samples_of_gt_ours_opencv.png", vis)

    print(f"PSNR --> Ours: {trained_psnr: .2f} | Best OpenCV: {max_cubic_psnr: .2f} | Avg OpenCV: {avg_cubic_psnr: .2f}")
    print(f"SSIM --> Ours: {trained_ssim: .2f} | Best OpenCV: {max_cubic_ssim: .2f} | Avg OpenCV: {avg_cubic_ssim: .2f} \n\n")
    return trained_psnr, max_cubic_psnr, avg_cubic_psnr, min_cubic_psnr, trained_ssim, max_cubic_ssim, avg_cubic_ssim, min_cubic_ssim

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="data/config.json", help="json config file")
    args = parser.parse_args()
    inference(args.config)
