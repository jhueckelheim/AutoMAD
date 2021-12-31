import torch

def channel2batch(x):
    # reinterpret a [x,y,:,:] tensor as a [x*y,1,:,:] tensor
    return x.view(x.size(1)*x.size(0), 1, x.size(2), x.size(3))

def truebatch2outer(x, n_batches):
    # reinterpret a [x*n_batches,:,:,:] tensor as a [n_batches,x,:,:,:] tensor
    return x.view(n_batches, x.size(0)//n_batches, x.size(1), x.size(2), x.size(3))

class DualTensor(torch.Tensor):
    '''
    DualTensor objects store a primal tensor and its corresponding derivative
    tensor. They otherwise inherit all functionality from torch.Tensor.
    DualTensor also contains a "gradient" tensor, which is really just a vector.
    During the forward pass it contains no useful information, and its length is
    the number of all trainable parameters for all previous layers and the
    current layer. The size of this vector is what is actually reported when the
    size() function is called for the DualTensor. This is done so that PyTorch
    will allow us to pass the gradient information to all layers during the
    backward sweep.
    '''
    @staticmethod
    def __new__(cls, xb, x, xd, *args, **kwargs):
        return super().__new__(cls, xb, *args, **kwargs)
    
    def __init__(self, xb, x, xd):
        '''
        x is expected to be the primal data, xd is the corresponding derivative data.
        If xd contains multiple directional derivatives, they are expected to be
        stored as channels. The number of channels in xd is the product of the
        number of channels in the primal data (the "true" channels) and the
        number of directional derivatives.
        '''
        super().__init__()
        self.x = x
        self.xd = xd

class Fwd2Rev(torch.nn.Module):
    '''
    A neural network layer whose only purpose is to glue forward-mode AD and
    reverse-mode AD together. It will store forward-propagated derivatives
    during the forward sweep, and combine them with the reverse-propagated
    derivatives during the reverse sweep, immediately resulting in the gradient
    tensors for all preceding forward layers.
    '''
    class __Func__(torch.autograd.Function):
        @staticmethod
        def forward(ctx, fwdinput):
            input_p = fwdinput.x
            input_d = fwdinput.xd
            ctx.save_for_backward(input_d)
            return input_p

        @staticmethod
        def backward(ctx, grad_output):
            input_d, = ctx.saved_tensors
            n_batches = grad_output.size(0)
            grad_input = (input_d*grad_output.unsqueeze(2)).sum(dim=[0,2,3,4])
            return grad_input.view(input_d.size(1))

    def __init__(self):
        super(Fwd2Rev, self).__init__()

    def forward(self, x):
        return self.__Func__.apply(x)

class Conv2d(torch.nn.Module):
    class __Func__(torch.autograd.Function):
        @staticmethod
        def forward(ctx, fwdinput_dual, weight, bias=None):
            with torch.no_grad():
                if(isinstance(fwdinput_dual, DualTensor)):
                    fwdinput = fwdinput_dual.x
                    fwdinput_d = fwdinput_dual.xd
                    num_dervs = fwdinput_dual.size(0)
                else:
                    fwdinput = fwdinput_dual
                    fwdinput_d = None
                    num_dervs = 0
                n_batches = fwdinput.size(0)
                if ctx.needs_input_grad[0]:
                    fwdinput_pd = torch.cat([fwdinput, fwdinput_d], dim=0)
                else:
                    fwdinput_pd = fwdinput
                # Primal Evaluation (includes weight*fwdinput_d)
                ret = torch.nn.functional.conv2d(fwdinput_pd, weight, bias)
                ret_p = ret[0:n_batches,:,:,:]
                ret_d1 = ret[n_batches:,:,:,:]
                # Derivative Propagation
                n_directions = 0
                if ctx.needs_input_grad[1]:
                    n_directions += 9
                if bias is not None and ctx.needs_input_grad[2]:
                    n_directions += 1
                bias_d = torch.nn.Parameter(torch.zeros(n_directions), requires_grad=False)
                weight_d = torch.nn.Parameter(torch.zeros(n_directions, 1, 3, 3), requires_grad=False)
                if ctx.needs_input_grad[1]:
                  weight_d[0:9,:,:,:].view(9*9)[0::10] = 1
                if bias is not None and ctx.needs_input_grad[2]:
                  bias_d[-1] = 1
                #ret_d2 = channel2batch(torch.nn.functional.conv2d(fwdinput, weight_d, bias_d))
                ret_d2 = channel2batch(torch.nn.functional.conv2d(fwdinput, weight_d, bias_d))
                if(ret_d1.size(0) > 0):
                    ret_d = torch.cat([truebatch2outer(ret_d1, n_batches), truebatch2outer(ret_d2, n_batches)], dim=1)
                else:
                    ret_d = ret_d2
            ret_dual = DualTensor(torch.zeros(10+num_dervs), ret_p, ret_d)
            return ret_dual

        @staticmethod
        def backward(ctx, grad_output):
            # grad_input is the pass-through of gradient information for layers
            # that happened before us in the forward pass (and will happen after
            # this in the reverse pass).
            # The gradient tensor is organized such that the last directional
            # derivatives belong to the current layer. This layer has 10
            # trainable parameters, so everything except the last 10 directions
            # is passed into grad_input.
            grad_input = grad_output[:-10]
            # the rest is gradient info for the current layer. The weights are
            # first, the bias is last.
            grad_weight = grad_output[-10:-1].view(1,1,3,3)
            grad_bias = grad_output[-1].view(1)
            return grad_input, grad_weight, grad_bias

    def __init__(self, in_channels, out_channels, kernel_size, bias = True):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # TODO we create a Conv2d layer here simply to piggy-back on its default initialization for weight and bias.
        # We should just find out whatever the correct default initialization is, and use that.
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias)
        self.weight = self.conv.weight
        self.bias = self.conv.bias

    def forward(self, x):
        return self.__Func__.apply(x, self.weight, self.bias)
