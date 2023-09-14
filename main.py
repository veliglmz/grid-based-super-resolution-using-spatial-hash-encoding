import argparse
import json
import os
import shutil
import pandas as pd
from inference import inference


def main(config_path):
    file = open(config_path)
    config = json.load(file)
    dataset_images_folder = sorted(os.listdir(config["dataset_path"]))
    comparison_csv = []
    for image_folder in dataset_images_folder:
        results_dir_path = os.path.join("results", image_folder)
        if os.path.exists(results_dir_path):
            shutil.rmtree(results_dir_path)
        os.mkdir(results_dir_path)

        trained_psnr, max_cubic_psnr, avg_cubic_psnr, min_cubic_psnr, trained_ssim, max_cubic_ssim, avg_cubic_ssim, min_cubic_ssim = inference(config, image_folder)
        
        if trained_psnr != 0:
            comparison_csv.append({"min_psnr":min_cubic_psnr, "avg_psnr":avg_cubic_psnr, "max_psnr":max_cubic_psnr, "ours_psnr":trained_psnr, 
                                "min_ssim":min_cubic_ssim, "avg_ssim":avg_cubic_ssim, "max_ssim":max_cubic_ssim, "ours_ssim":trained_ssim})

    comparison_csv = pd.DataFrame(comparison_csv)
    mean_min_psnr = comparison_csv["min_psnr"].mean()
    mean_avg_psnr = comparison_csv["avg_psnr"].mean()
    mean_max_psnr = comparison_csv["max_psnr"].mean()
    mean_ours_psnr = comparison_csv["ours_psnr"].mean()

    mean_min_ssim = comparison_csv["min_ssim"].mean()
    mean_avg_ssim = comparison_csv["avg_ssim"].mean()
    mean_max_ssim = comparison_csv["max_ssim"].mean()
    mean_ours_ssim = comparison_csv["ours_ssim"].mean()

    print(f"PSNR | Mean of Min. : {mean_min_psnr:.2f}")
    print(f"PSNR | Mean of Avg. : {mean_avg_psnr:.2f}")
    print(f"PSNR | Mean of Max. : {mean_max_psnr:.2f}")
    print(f"PSNR | Mean of Ours : {mean_ours_psnr:.2f}")
    print()
    print(f"SSIM | Mean of Min. : {mean_min_ssim:.2f}")
    print(f"SSIM | Mean of Avg. : {mean_avg_ssim:.2f}")
    print(f"SSIM | Mean of Max. : {mean_max_ssim:.2f}")
    print(f"SSIM | Mean of Ours : {mean_ours_ssim:.2f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config.json", help="Config file path.")
    args = parser.parse_args()
    main(args.config)
