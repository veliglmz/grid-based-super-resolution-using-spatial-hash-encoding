import math

import torch
from numba import cuda
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function


torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)


def determine_activation(activation_name):
    if activation_name == "ReLU":
        return nn.ReLU()
    elif activation_name == "Leaky_ReLU":
        return nn.LeakyReLU()
    elif activation_name == "Tanh":
        return nn.Tanh()
    elif activation_name == "GELU":
        return nn.GELU()
    else:
        print(f"Activation is not defined! --> {activation_name}")
        return None


def determine_optimizer(optimizer_type, network_params, learning_rate):
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(network_params, lr=learning_rate, betas=(0.9, 0.99), eps=1e-15, weight_decay=1e-06)
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(network_params, lr=learning_rate)
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(network_params, lr=learning_rate)
    else:
        optimizer = None
        print("Optimizer is None!!!")
    return optimizer


def determine_criterion(loss_type):
    if loss_type == "RelativeL2":
        criterion = torch.nn.MSELoss()
    elif loss_type == "L1Loss":
        criterion = torch.nn.L1Loss()
    elif loss_type == "SmoothL1":
        criterion = torch.nn.SmoothL1Loss()
    else:
        criterion = None
        print("Loss function is None!!!")
    return criterion


def calculate_xs_and_ys(width, height, n_coords_padded):
    """
    For each pixel, we calculate x and y coordinates (+0.5 for truncating) and normalize them.
    """
    xs_and_ys = np.zeros((n_coords_padded * 2,), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            idx = (y * width + x) * 2
            xs_and_ys[idx] = float(x + 0.5) / float(width)
            xs_and_ys[idx + 1] = float(y + 0.5) / float(height)
    return xs_and_ys


def read_params(file_path):
    params = []
    fr = open(file_path, "r")
    data = fr.readlines()
    for line in data:
        params.append(float(line.strip()))
    return np.asarray(params, dtype=np.float32)


def div_round_up(val, div):
    return int(((val + div - 1) / div))


def next_multiple(val, div):
    """
    For using GPU efficiently
    """
    dru = div_round_up(val, div)
    res = int(dru * div)
    return res


@cuda.jit(inline=True)
def determine_xy0_cuda(xi, yi, width, height):
    if xi < 0:
        x0 = 0
    elif xi > width - 1:
        x0 = width - 1
    else:
        x0 = xi

    if yi < 0:
        y0 = 0
    elif yi > height - 1:
        y0 = height - 1
    else:
        y0 = yi

    return int(x0), int(y0)


@cuda.jit(inline=True)
def determine_xy1_cuda(x0, y0, width, height):
    if x0 + 1 < 0:
        x1 = 0
    elif x0 + 1 > width - 1:
        x1 = width - 1
    else:
        x1 = x0 + 1

    if y0 + 1 < 0:
        y1 = 0
    elif y0 + 1 > height - 1:
        y1 = height - 1
    else:
        y1 = y0 + 1
    return int(x1), int(y1)



def determine_xy0(xi, yi, width, height):
    if xi < 0:
        x0 = 0
    elif xi > width - 1:
        x0 = width - 1
    else:
        x0 = xi

    if yi < 0:
        y0 = 0
    elif yi > height - 1:
        y0 = height - 1
    else:
        y0 = yi

    return int(x0), int(y0)


def determine_xy1(x0, y0, width, height):
    if x0 + 1 < 0:
        x1 = 0
    elif x0 + 1 > width - 1:
        x1 = width - 1
    else:
        x1 = x0 + 1

    if y0 + 1 < 0:
        y1 = 0
    elif y0 + 1 > height - 1:
        y1 = height - 1
    else:
        y1 = y0 + 1
    return int(x1), int(y1)


@cuda.jit(inline=True)
def fast_hash(x, y, hashmap_size):
    index = 0
    index ^= x * 1
    index ^= y * 2654435761
    index = (index % hashmap_size)
    return int(index)


@cuda.jit(inline=True)
def calculate_filter_mode_linear_cuda(o1, o2, o3, o4, lwx, lwy):
    return (o1 * (1.0 - lwx) * (1.0 - lwy) +
            o2 * lwx * (1.0 - lwy) +
            o3 * (1.0 - lwx) * lwy +
            o4 * lwx * lwy)

def calculate_filter_mode_linear(o1, o2, o3, o4, lwx, lwy):
    return (o1 * (1.0 - lwx) * (1.0 - lwy) +
            o2 * lwx * (1.0 - lwy) +
            o3 * (1.0 - lwx) * lwy +
            o4 * lwx * lwy)


@cuda.jit
def relu(v):
    if v > 0.0:
        return v
    else:
        return 0.0


@cuda.jit()
def _forward(xs_and_ys, n_elements, hashmap_offsets_table, n_features, log2_per_level_scale, base_resolution,
             grid_params, n_levels, forward_output, n_backward_contents, backward_output, derivative):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    level = cuda.blockIdx.y  # < - the level is the same for all threads
    if i >= n_elements or level >= n_levels:
        return

    grid_index = int(hashmap_offsets_table[level] * n_features)
    hashmap_size = int(hashmap_offsets_table[level + 1]) - int(hashmap_offsets_table[level])

    scale = math.pow(2, level * math.log2(log2_per_level_scale)) * base_resolution - 1.0
    scale = round(scale, 10)
    resolution = int(math.ceil(scale)) + 1

    pos_x = xs_and_ys[i * 2] * scale + 0.5
    pos_y = xs_and_ys[i * 2 + 1] * scale + 0.5

    pos_grid_x = math.floor(pos_x)
    pos_grid_y = math.floor(pos_y)

    lerp_weight_x = pos_x - pos_grid_x
    lerp_weight_y = pos_y - pos_grid_y

    x0, y0 = determine_xy0_cuda(pos_grid_x, pos_grid_y, resolution, resolution)
    x1, y1 = determine_xy1_cuda(x0, y0, resolution, resolution)
    i1 = x0 + y0 * resolution
    i2 = x1 + y0 * resolution
    i3 = x0 + y1 * resolution
    i4 = x1 + y1 * resolution

    if math.pow(resolution, 2) > hashmap_size:
        i1 = fast_hash(x0, y0, hashmap_size)
        i2 = fast_hash(x1, y0, hashmap_size)
        i3 = fast_hash(x0, y1, hashmap_size)
        i4 = fast_hash(x1, y1, hashmap_size)

    for f in range(n_features):
        g1 = grid_params[grid_index + (i1 * n_features) + f]
        g2 = grid_params[grid_index + (i2 * n_features) + f]
        g3 = grid_params[grid_index + (i3 * n_features) + f]
        g4 = grid_params[grid_index + (i4 * n_features) + f]

        features = calculate_filter_mode_linear_cuda(g1, g2, g3, g4, lerp_weight_x, lerp_weight_y)

        forward_output[(i * n_features * n_levels) + (level * n_features) + f] = features

    if derivative:
        d1 = (1.0 - lerp_weight_x) * (1.0 - lerp_weight_y)
        d2 = lerp_weight_x * (1.0 - lerp_weight_y)
        d3 = (1.0 - lerp_weight_x) * lerp_weight_y
        d4 = lerp_weight_x * lerp_weight_y

        backward_output[(i * n_levels * n_backward_contents) + (level * n_backward_contents) + 0] = i1
        backward_output[(i * n_levels * n_backward_contents) + (level * n_backward_contents) + 1] = i2
        backward_output[(i * n_levels * n_backward_contents) + (level * n_backward_contents) + 2] = i3
        backward_output[(i * n_levels * n_backward_contents) + (level * n_backward_contents) + 3] = i4

        backward_output[(i * n_levels * n_backward_contents) + (level * n_backward_contents) + 4] = grid_index

        backward_output[(i * n_levels * n_backward_contents) + (level * n_backward_contents) + 5] = d1
        backward_output[(i * n_levels * n_backward_contents) + (level * n_backward_contents) + 6] = d2
        backward_output[(i * n_levels * n_backward_contents) + (level * n_backward_contents) + 7] = d3
        backward_output[(i * n_levels * n_backward_contents) + (level * n_backward_contents) + 8] = d4


@cuda.jit()
def _sum_update_grid_params(n_elements, inputs_grad, encoding_backward_output, n_features, sum_updated_grid_params):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= n_elements:
        return
    l = cuda.blockIdx.y
    i1, i2, i3, i4, grid_index, d1, d2, d3, d4 = encoding_backward_output[i][l]

    for f in range(n_features):
        input_grad = inputs_grad[i][l * n_features + f]

        sum_updated_grid_params[int(grid_index + (i1 * n_features) + f)] += input_grad * d1
        sum_updated_grid_params[int(grid_index + (i2 * n_features) + f)] += input_grad * d2
        sum_updated_grid_params[int(grid_index + (i3 * n_features) + f)] += input_grad * d3
        sum_updated_grid_params[int(grid_index + (i4 * n_features) + f)] += input_grad * d4


@cuda.jit()
def _update_grid_params(n_elements, sum_updated_grid_params, lr, grid_params, n_features):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= n_elements:
        return
    # TODO: remove n_features
    grid_params[i] -= (sum_updated_grid_params[i] * lr) #/ n_features) * lr


@cuda.jit(inline=True)
def _determine_input_samples_cuda(batch_size, random_coords, xs_and_ys, inputs_xs_and_ys):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= batch_size:
        return

    r = random_coords[i]

    inputs_xs_and_ys[i * 2] = xs_and_ys[r * 2]
    inputs_xs_and_ys[i * 2 + 1] = xs_and_ys[r * 2 + 1]


@cuda.jit(inline=True)
def _determine_gt_samples_cuda(batch_size, random_coords, scale_factor, w, h, gt, img, tx, ty, interpolation_type):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= batch_size:
        return

    r = random_coords[i]

    iy = ((r // w))
    ix = ((r %  w))

    gt[i * 3 + 0] = img[iy][ix][0]
    gt[i * 3 + 1] = img[iy][ix][1]
    gt[i * 3 + 2] = img[iy][ix][2]


    

@cuda.jit()
def _determine_gt_samples_bilinear(i, img, ix, iy, w, h, gt):
    pos_grid_x = math.floor(ix)
    pos_grid_y = math.floor(iy)

    lerp_weight_x = ix - pos_grid_x
    lerp_weight_y = iy - pos_grid_y

    x0, y0 = determine_xy0_cuda(pos_grid_x, pos_grid_y, w, h)
    x1, y1 = determine_xy1_cuda(x0, y0, w, h)

    p1 = img[y0][x0]
    p2 = img[y0][x1]
    p3 = img[y1][x0]
    p4 = img[y1][x1]

    for j in range(3):
        p = calculate_filter_mode_linear_cuda(p1[j], p2[j], p3[j], p4[j], lerp_weight_x, lerp_weight_y)
        gt[i * 3 + j] = p


@cuda.jit()
def W_cuda(x):
    '''Weight function that return weight for each distance point
    Parameters:
    x (float): Distance from destination point
    Returns:
    float: Weight
    '''
    a = -0.5
    pos_x = abs(x)
    if -1 <= abs(x) <= 1:
        return ((a+2)*(pos_x**3)) - ((a+3)*(pos_x**2)) + 1
    elif 1 < abs(x) < 2 or -2 < x < -1:
        return ((a * (pos_x ** 3)) - (5 * a * (pos_x ** 2)) + (8 * a * pos_x) - 4 * a)
    else:
        return 0


@cuda.jit()
def _determine_gt_samples_bicubic(i, img, xm, ym, gt):
    xi = math.floor(xm)
    yi = math.floor(ym)

    u = xm - xi
    v = ym - yi

    for c in range(3):
        out = 0
        for n in range(-1, 3):
            for m in range(-1, 3):
                if (xi + n < 0) or (xi + n >= img.shape[1]) or (yi + m < 0) or (yi + m >= img.shape[0]):
                    continue

                p = img[int(yi + m)][int(xi + n)][c]
                Wu = W_cuda(u - n)
                Wv = W_cuda(v - m)
                out = out + (p * (Wu * Wv))

        if out < 0:
            gt[i * 3 + c] = 0
        elif out > 1.0:
            gt[i * 3 + c] = 1.0
        else:
            gt[i * 3 + c] = out


