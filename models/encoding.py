import numpy as np
from numba import cuda
import math
from utils.model_utils import next_multiple, div_round_up, _forward, _sum_update_grid_params, _update_grid_params

np.random.seed(42)


class Encoding:
    def __init__(self, scale, n_levels, base_resolution, n_features, hashmap_size):
        self.scale = scale
        self.n_levels = n_levels
        self.base_resolution = base_resolution
        self.n_features = n_features
        self.hashmap_size = hashmap_size
        self.n_backward_contents = 9  # 4 indices, grid index, and 4 derivatives
        self.hashmap_offsets_table, self.n_params = self.determine_grids()
        self.grid_params = np.full((self.n_params,), fill_value=np.random.uniform(-1e-4, 1e-4), dtype=np.float32)

    def determine_grids(self):
        hashmap_offsets_table = np.empty((self.n_levels + 1,), dtype=np.float32)
        offset = 0
        max_params = np.Inf
        for i in range(self.n_levels):
            scale = math.pow(2, i * math.log2(self.scale)) * self.base_resolution - 1.0
            # to eliminate the floating point rounding
            scale = round(scale, 10)
            resolution = int(math.ceil(scale)) + 1
            params_in_level = max_params if np.power(resolution, 2) > max_params else np.power(resolution, 2)
            # it is not allowed to exceed the hashmap size
            params_in_level = min(params_in_level, self.hashmap_size)
            params_in_level = next_multiple(params_in_level, 8)

            hashmap_offsets_table[i] = offset
            offset += params_in_level

        hashmap_offsets_table[self.n_levels] = offset
        n_params = int(hashmap_offsets_table[self.n_levels] * self.n_features)
        return hashmap_offsets_table, n_params

    def load_grid_params(self, hash_params):
        hash_index = 0
        for l in range(self.n_levels):
            offset = int(self.hashmap_offsets_table[l + 1] - self.hashmap_offsets_table[l])
            for i in range(offset * 2):
                self.grid_params[hash_index] = hash_params[hash_index]
                hash_index += 1

    def forward(self, xs_and_ys, num_elements, is_inference=False):
        # This two lines must be on device because we use them for extracting encodings
        # however we cannot define it before because we have two options to initialization of grid parameters.
        # first one is randomly and another is load_grid_params method.
        self.hashmap_offsets_table = cuda.to_device(self.hashmap_offsets_table)
        self.grid_params = cuda.to_device(self.grid_params)

        forward_output = cuda.device_array((num_elements * self.n_levels * self.n_features,))
        xs_and_ys = cuda.to_device(xs_and_ys)

        # we want to execute this function for all pixels given xs_and_ys for each level.
        threads_per_block = 512
        x = div_round_up(num_elements, threads_per_block)
        blocks_hash_grid = [x, self.n_levels, 1]

        if is_inference:
            backward_output = cuda.device_array(0)
            _forward[blocks_hash_grid, threads_per_block](xs_and_ys, num_elements, self.hashmap_offsets_table,
                                                          self.n_features, self.scale, self.base_resolution,
                                                          self.grid_params, self.n_levels, forward_output,
                                                          self.n_backward_contents, backward_output, 0)
        else:
            backward_output = cuda.device_array((num_elements * self.n_levels * self.n_backward_contents,))
            _forward[blocks_hash_grid, threads_per_block](xs_and_ys, num_elements, self.hashmap_offsets_table,
                                                          self.n_features, self.scale, self.base_resolution,
                                                          self.grid_params, self.n_levels, forward_output,
                                                          self.n_backward_contents, backward_output, 1)
        return forward_output, backward_output

    def update_grid_params(self, num_elements, inputs_grad, encoding_backward_output, lr):
        """
        Backward processes.
        """
        threads_per_block = 512
        x = div_round_up(num_elements, threads_per_block)
        blocks_hash_grid = [x, self.n_levels, 1]

        inputs_grad = cuda.to_device(inputs_grad)
        encoding_backward_output = cuda.to_device(encoding_backward_output)
        sum_updated_grid_params = cuda.device_array((self.n_params,))

        # it calculates the sum of grids.
        _sum_update_grid_params[blocks_hash_grid, threads_per_block](num_elements, inputs_grad,
                                                                     encoding_backward_output, self.n_features,
                                                                     sum_updated_grid_params)
        # it updates the grid parameters.
        threads_per_block = 512
        x = div_round_up(self.n_params, threads_per_block)
        blocks_hash_grid = [x, 1]
        _update_grid_params[blocks_hash_grid, threads_per_block](self.n_params, sum_updated_grid_params, lr,
                                                                 self.grid_params, self.n_features)
