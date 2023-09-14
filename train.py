import random
from tqdm import tqdm
import numpy as np
from timeit import default_timer as timer
import torch
from numba import cuda

from utils.image_utils import write_image
from utils.model_utils import div_round_up, _determine_input_samples_cuda, _determine_gt_samples_cuda


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


class Trainer:
    def __init__(self, encoding, network, optimizer, criterion, n_epochs, xs_and_ys,
                 n_coords, batch_size, scale_factor, width, height, imgs,
                 interpolation_type, result_dir_path):
        self.interpolation_dict = {"bilinear": 0, "bicubic": 1}
        self.encoding = encoding
        self.network = network.cuda()
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 100000, gamma=0.8)
        self.criterion = criterion
        self.n_epochs = n_epochs
        self.imgs = imgs
        self.interpolation_type = self.interpolation_dict[interpolation_type]
        self.scale_factor = scale_factor


        self.xs_and_ys = xs_and_ys
        self.n_coords = n_coords
        self.batch_size = 1 << batch_size
        self.width = width
        self.height = height
        self.result_dir_path = result_dir_path

    def train(self):
        n_levels = self.encoding.n_levels
        n_features = self.encoding.n_features
        self.network.train()
        # we only select pixels as many as the number of batch size for each epoch.
        h, w = self.imgs[0][0].shape[:2]
        best_loss = np.inf
      
        for epoch in tqdm(range(1, self.n_epochs + 1)):

            random_coords = []
            for _ in range(self.batch_size):
                random_coords.append(random.randint(0, self.n_coords-1))
            random_coords = np.asarray(random_coords, dtype=np.int64)

            inputs_xs_and_ys, gts = self._determine_samples(random_coords)

            encoding_forward_output, encoding_backward_output = self.encoding.forward(inputs_xs_and_ys, self.batch_size)
            encoding_forward_output = encoding_forward_output.copy_to_host()
            encoding_backward_output = encoding_backward_output.copy_to_host()
            # reshape the outputs of the encoding to give them to the network.
            encoding_forward_output = encoding_forward_output.reshape(-1, n_levels * n_features).astype(np.float32)
            encoding_backward_output = encoding_backward_output.reshape(-1, n_levels, self.encoding.n_backward_contents)\
                                                                        .astype(np.float32)

            self.optimizer.zero_grad()
            encoding_outputs = torch.from_numpy(encoding_forward_output).cuda()
            # to update the grid params, we need to derivatives of the input layer.
            encoding_outputs.requires_grad = True

            translated_random_coords_x = []
            translated_random_coords_y = []

            for _, t in self.imgs:
                random_coords_y_ = np.full((encoding_outputs.size(0),), t[1])
                random_coords_x_ = np.full((encoding_outputs.size(0),), t[0])

                translated_random_coords_x.append(random_coords_x_)
                translated_random_coords_y.append(random_coords_y_)

            translated_random_coords_x = np.asarray(translated_random_coords_x, dtype=np.float32)
            translated_random_coords_y = np.asarray(translated_random_coords_y, dtype=np.float32)
            
            translated_random_coords_x = torch.from_numpy(translated_random_coords_x).cuda()
            translated_random_coords_y = torch.from_numpy(translated_random_coords_y).cuda()

            # the outputs convolutions with size of batch
            network_outputs = self.network(encoding_outputs, translated_random_coords_x, translated_random_coords_y, w, h)
            
            gts_torch = []
            for i, gt in enumerate(gts):
                gt = gt.copy_to_host()
                gt = gt.reshape(-1, 3).astype(np.float32)
                gts_torch.append(gt)
            
            gts_torch = np.asarray(gts_torch, dtype=np.float32)
            gts_torch = torch.from_numpy(gts_torch).cuda()

            
            loss = self.criterion(network_outputs, gts_torch)
            loss.backward()
            loss = loss.item()
            self.optimizer.step()
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]
            inputs_grad = self.network.inputs.grad
            
            # backward for grids
            # we need to add interpolation and H_i modules derivatives here.
            self.encoding.update_grid_params(self.batch_size, inputs_grad, encoding_backward_output, lr)
            #print(f"Epoch {epoch} | # of samples: {self.batch_size} | Loss: {loss: .5f}")

            if loss < best_loss:
                #print(f"Best loss: {loss}\n")
                best_loss = loss
                self.inference(name="best")
                self.network.train()

            
    def inference(self, name=""):
        n_levels = self.encoding.n_levels
        n_features = self.encoding.n_features
        h, w = self.imgs[0][0].shape[:2]
        self.network.eval()

        with torch.no_grad():
            encoding_forward_output, _ = self.encoding.forward(self.xs_and_ys, self.n_coords, is_inference=True)
            encoding_forward_output = encoding_forward_output.copy_to_host()
            encoding_forward_output = encoding_forward_output.reshape(-1, n_levels * n_features).astype(np.float32)

            encoding_outputs = torch.from_numpy(encoding_forward_output).cuda()
            
            center_x = torch.full((1, encoding_outputs.size(0)), 0.0).cuda()
            center_y = torch.full((1, encoding_outputs.size(0)), 0.0).cuda()

            network_output = self.network(encoding_outputs, center_x, center_y, w, h)
            network_output = network_output.detach().cpu().numpy()[0]

            output_img = np.reshape(network_output, (int(h), int(w), 3))

            output_img[output_img > 1.0] = 1.0
            output_img[output_img < 0.0] = 0.0
            write_image(f"{self.result_dir_path}/inferences/inference_{name}.png", output_img)


    def _determine_samples(self, random_coords):
        for i in range(len(self.imgs) - 1):
            assert self.imgs[i][0].shape == self.imgs[i+1][0].shape, "Source images must be the same dimensionality."
        
        random_coords = cuda.to_device(random_coords)

        h, w = self.imgs[0][0].shape[:2]

        xs_and_ys = cuda.to_device(self.xs_and_ys)

        inputs_xs_and_ys = cuda.device_array((self.batch_size * 2,))

        threads_per_block = 32
        x = div_round_up(self.batch_size, threads_per_block)
        blocks_hash_grid = [x, 1]
        _determine_input_samples_cuda[blocks_hash_grid, threads_per_block](self.batch_size, random_coords,
                                                                           xs_and_ys, inputs_xs_and_ys)
        inputs_xs_and_ys = inputs_xs_and_ys.copy_to_host()

        gts = []
        
        for img, t in self.imgs:
            gt = cuda.device_array((self.batch_size * 3))
            img = cuda.to_device(img)

            threads_per_block = 16
            x = div_round_up(self.batch_size, threads_per_block)
            blocks_hash_grid = [x, 1]
            _determine_gt_samples_cuda[blocks_hash_grid, threads_per_block](self.batch_size, random_coords, 
                                                                            self.scale_factor, w, h, gt, img, t[0], t[1],
                                                                            self.interpolation_type)
            gt.copy_to_host()
            gts.append(gt)
        return inputs_xs_and_ys, gts
