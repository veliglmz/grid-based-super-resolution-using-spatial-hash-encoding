import cv2
from PIL import Image
import numpy as np
import torch
import os
import pickle
import json
from datasets.synthetic_burst_train_set import SyntheticBurst
from utils.postprocessing_functions import SimplePostProcess
from utils.warp import warp
import argparse

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = '''Dataset Generation''',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-t', '--type',
                        required=True,
                        help = 'MFSR or SISR')
    parser.add_argument('-ns', '--number_of_samples',
                        default=40,
                        type=int,
                        help = 'the number of samples for each image')
    parser.add_argument('-ni', '--number_of_images',
                        default=14,
                        type=int,
                        help = 'the number of images to generate a high-resolution image')

    args = parser.parse_args()

    if args.type == "MFSR":

        deep_dataset = "our_dataset_for_deeprep"
        deep_bursts = f"{deep_dataset}/bursts"
        deep_gt = f"{deep_dataset}/gt"
        os.makedirs(deep_bursts, exist_ok=True)
        os.makedirs(deep_gt, exist_ok=True)

        our_dataset = "our_dataset_for_ours"

        img_folder = "original_images"
        img_paths = os.listdir(img_folder)

        c = 0

        for img_path in img_paths:
            img_path = os.path.join(img_folder, img_path)
            
            for _ in range(args.number_of_samples):
                
                num = "{0:04d}".format(c)
                print(num)
                deep_burst_img = f"{deep_bursts}/{num}"
                deep_gt_img = f"{deep_gt}/{num}"
                os.makedirs(deep_burst_img, exist_ok=True)
                os.makedirs(deep_gt_img, exist_ok=True)

                our_img = f"{our_dataset}/{num}"
                our_lrs = f"{our_img}/lrs"
                our_warped_lrs = f"{our_img}/warped_lrs"
                os.makedirs(our_img, exist_ok=True)
                os.makedirs(our_lrs, exist_ok=True)
                os.makedirs(our_warped_lrs, exist_ok=True)


                base_dataset = [Image.open(img_path)]
                burst_process_fn = SimplePostProcess(return_np=True, demosaics=False)
                gt_process_fn = SimplePostProcess(return_np=True)
                synt = SyntheticBurst(base_dataset=base_dataset)

                
                for d in synt:
                    burst, frame_gt, burst_rgb, flow_vectors, meta_info = d
                    
                    # DeepRep GT IMAGE AND META INFO
                    deeprep_gt = frame_gt.cpu().detach().numpy()
                    deeprep_gt = np.moveaxis(deeprep_gt, [0], [2])

                    deeprep_gt = np.clip(deeprep_gt, 0.0, 1.0) * 2 ** 14
                    deeprep_gt = deeprep_gt.astype(np.uint16)
                    cv2.imwrite(f"{deep_gt_img}/im_rgb.png", deeprep_gt)
                    with open(f"{deep_gt_img}/meta_info.pkl", 'wb') as handle:
                        pickle.dump(meta_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    # OUR GT IMAGE
                    frame_gt = gt_process_fn.process(frame_gt, meta_info)
                    cv2.imwrite(f"{our_img}/hr.png", frame_gt)

                    flow_vectors = -flow_vectors + 0.5
                    translations = []
                    for i in range(args.number_of_images):
                        
                        # DEEPREP BURST IMAGE
                        deep_burst = burst[i].detach().cpu().numpy()
                        deep_burst = np.moveaxis(deep_burst, [0], [2])
                        deep_burst = np.clip(deep_burst, 0.0, 1.0) * 2 ** 14
                        deep_burst = deep_burst.astype(np.uint16)
                        num = "{0:02d}".format(i)
                        cv2.imwrite(f"{deep_burst_img}/im_raw_{num}.png", deep_burst)

                        burst_rgb0 = burst_rgb[i].unsqueeze(0)
                        burst_rgb0 = burst_process_fn.process(burst_rgb0[0], meta_info).astype(np.float32)
                        burst_rgb0 = torch.from_numpy(burst_rgb0)
                        burst_rgb0 = burst_rgb0.permute(2, 0, 1).unsqueeze(0)

                        ft = -flow_vectors[i].unsqueeze(0)
                        warped_brst_rgb0 = warp(burst_rgb0, -ft)
                        warped_brst_rgb0 = warped_brst_rgb0[0]
                        warped_brst_rgb0 = warped_brst_rgb0.permute(1, 2, 0).numpy().astype(np.uint8)
                        cv2.imwrite(f"{our_warped_lrs}/lr{i}.png", warped_brst_rgb0)

                        burst_rgb0 = burst_rgb0[0]
                        burst_rgb0 = burst_rgb0.permute(1, 2, 0).numpy().astype(np.uint8)
                        cv2.imwrite(f"{our_lrs}/lr{i}.png", burst_rgb0)

                        tx = flow_vectors[i][0]
                        ty = flow_vectors[i][1]
                        translations.append([tx.mean(0).mean().item(), ty.mean(0).mean().item()])
                    
                    info_json = {"scale_factor": 4, "translations": translations}
                    json_object = json.dumps(info_json)

                    with open(f"{our_img}/info.json", "w") as outfile:
                        outfile.write(json_object)
                c += 1

    elif args.type == "SISR":

        our_dataset = "our_dataset_for_sisr"
        img_folder = "original_images"
        img_paths = os.listdir(img_folder)

        c = 0
        for img_path in img_paths:
            img_path = os.path.join(img_folder, img_path)
            
            for _ in range(args.number_of_samples):
                num = "{0:04d}".format(c)
                print(num)
                our_img = f"{our_dataset}/{num}"
                our_lrs = f"{our_img}/lrs"
                our_warped_lrs = f"{our_img}/warped_lrs"
                os.makedirs(our_img, exist_ok=True)
                os.makedirs(our_lrs, exist_ok=True)
                os.makedirs(our_warped_lrs, exist_ok=True)

                base_dataset = [Image.open(img_path)]

                burst_process_fn = SimplePostProcess(return_np=True, demosaics=False)
                gt_process_fn = SimplePostProcess(return_np=True)
                synt = SyntheticBurst(base_dataset=base_dataset, downsample_factor=8, crop_sz=512)

                for d in synt:
                    burst, frame_gt, burst_rgb, flow_vectors, meta_info = d
                    
                    # OUR GT IMAGE
                    frame_gt = gt_process_fn.process(frame_gt, meta_info)
                    cv2.imwrite(f"{our_img}/hr.png", frame_gt)

                    flow_vectors = -flow_vectors + 0.5
                    translations = []
                    for i in range(args.number_of_images):

                        burst_rgb0 = burst_rgb[i].unsqueeze(0)
                        burst_rgb0 = burst_process_fn.process(burst_rgb0[0], meta_info).astype(np.float32)
                        burst_rgb0 = torch.from_numpy(burst_rgb0)
                        burst_rgb0 = burst_rgb0.permute(2, 0, 1).unsqueeze(0)

                        ft = -flow_vectors[i].unsqueeze(0)
                        warped_brst_rgb0 = warp(burst_rgb0, -ft)
                        warped_brst_rgb0 = warped_brst_rgb0[0]
                        warped_brst_rgb0 = warped_brst_rgb0.permute(1, 2, 0).numpy().astype(np.uint8)
                        cv2.imwrite(f"{our_warped_lrs}/lr{i}.png", warped_brst_rgb0)

                        burst_rgb0 = burst_rgb0[0]
                        burst_rgb0 = burst_rgb0.permute(1, 2, 0).numpy().astype(np.uint8)
                        cv2.imwrite(f"{our_lrs}/lr{i}.png", burst_rgb0)

                        tx = flow_vectors[i][0]
                        ty = flow_vectors[i][1]
                        translations.append([tx.mean(0).mean().item(), ty.mean(0).mean().item()])
                    
                    info_json = {"scale_factor": 8, "translations": translations}
                    json_object = json.dumps(info_json)

                    with open(f"{our_img}/info.json", "w") as outfile:
                        outfile.write(json_object)
                c += 1
    
    else:
        print("Not Implemented.")
                    
                
