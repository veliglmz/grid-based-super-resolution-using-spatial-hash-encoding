import torch
import torch.nn as nn
from utils.model_utils import determine_activation


torch.set_printoptions(precision=8)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


class Bilinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, convs, ixs, iys, width, height):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        
        """

        n_imgs = ixs.size(0)

        # !!! translation must be at most 0.5 or at least -0.5
        ixs = (ixs + 0.5).clamp(0.0, 1.0)
        iys = (iys + 0.5).clamp(0.0, 1.0)
        
        pos_grid_x = torch.div(ixs, 1, rounding_mode="floor")
        pos_grid_y = torch.div(iys, 1, rounding_mode="floor")

        lwx = ixs - pos_grid_x
        lwy = iys - pos_grid_y
        
        b =  convs[:, 0, 0, 0] * (1.0 - lwx) * (1.0 - lwy) + \
             convs[:, 0, 0, 1] * lwx * (1.0 - lwy) + \
             convs[:, 0, 1, 0] * (1.0 - lwx) * lwy + \
             convs[:, 0, 1, 1] * lwx * lwy
        
        g =  convs[:, 1, 0, 0]  * (1.0 - lwx) * (1.0 - lwy) + \
             convs[:, 1, 0, 1]  * lwx * (1.0 - lwy) + \
             convs[:, 1, 1, 0]  * (1.0 - lwx) * lwy + \
             convs[:, 1, 1, 1]  * lwx * lwy

        r =  convs[:, 2, 0, 0]  * (1.0 - lwx) * (1.0 - lwy) + \
             convs[:, 2, 0, 1]  * lwx * (1.0 - lwy) + \
             convs[:, 2, 1, 0]  * (1.0 - lwx) * lwy + \
             convs[:, 2, 1, 1]  * lwx * lwy


        results = torch.stack((b, g, r), -1)

        w1 = (1.0 - lwx) * (1.0 - lwy)
        w2 = lwx * (1.0 - lwy)
        w3 = (1.0 - lwx) * lwy
        w4 = lwx * lwy

        w1 = torch.stack((w1, w1, w1), -1)
        w2 = torch.stack((w2, w2, w2), -1)
        w3 = torch.stack((w3, w3, w3), -1)
        w4 = torch.stack((w4, w4, w4), -1)

        w = torch.stack((w1, w2, w3, w4), -1).view(n_imgs, -1, 3, 2, 2)

        ctx.save_for_backward(w)

        return results


    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        w, = ctx.saved_tensors
        w[:, :, 0, 0, 0] = grad_output[:, :,  0] * w[:, :, 0, 0, 0]
        w[:, :, 0, 0, 1] = grad_output[:, :,  0] * w[:, :, 0, 0, 1]
        w[:, :, 0, 1, 0] = grad_output[:, :,  0] * w[:, :, 0, 1, 0]
        w[:, :, 0, 1, 1] = grad_output[:, :,  0] * w[:, :, 0, 1, 1]
        
        w[:, :, 1, 0, 0] = grad_output[:, :,  1] * w[:, :, 1, 0, 0]
        w[:, :, 1, 0, 1] = grad_output[:, :,  1] * w[:, :, 1, 0, 1]
        w[:, :, 1, 1, 0] = grad_output[:, :,  1] * w[:, :, 1, 1, 0]
        w[:, :, 1, 1, 1] = grad_output[:, :,  1] * w[:, :, 1, 1, 1]
        
        w[:, :, 2, 0, 0] = grad_output[:, :,  2] * w[:, :, 2, 0, 0]
        w[:, :, 2, 0, 1] = grad_output[:, :,  2] * w[:, :, 2, 0, 1]
        w[:, :, 2, 1, 0] = grad_output[:, :,  2] * w[:, :, 2, 1, 0]
        w[:, :, 2, 1, 1] = grad_output[:, :,  2] * w[:, :, 2, 1, 1]
        w = torch.sum(w, dim=0)
        return w, None, None, None, None


class Network(nn.Module):
    def __init__(self, activation_name, n_input, n_neurons, n_hidden_layers):
        super().__init__()
        self.activation = determine_activation(activation_name)
        self.n_input = n_input
        self.n_neurons = n_neurons
        self.n_hidden_layers = n_hidden_layers
        self.n_output =  2 * 2 * 3

        self.inputs = None  # we add this variable for differentiation.
        self.inputs_layer = nn.Linear(n_input, n_neurons, bias=False)# bias=True)
        self.hidden_layers = nn.Sequential()
        #self.dropout = nn.Dropout(p=0.2)

        for _ in range(n_hidden_layers):
            self.hidden_layers.append(nn.Linear(n_neurons, n_neurons, bias=False))# bias=True)
            self.hidden_layers.append(self.activation)
            #self.hidden_layers.append(self.dropout)

        self.output_layer = nn.Linear(self.n_neurons, self.n_output, bias=False) # bias=True)

        self.bilinear = Bilinear.apply

    def forward(self, x, translated_batched_coords_x, translated_batched_coords_y, width, height):
        self.inputs = x
        x = self.inputs_layer(self.inputs)
        x = self.activation(x)
        #x = self.dropout(x)

        x = self.hidden_layers(x)
        
        x = self.output_layer(x)

        x = self.activation(x)
    
        convs = x.view(-1, 3, 2, 2)

        results = self.bilinear(convs, translated_batched_coords_x, translated_batched_coords_y, width, height) 

        return results
        


