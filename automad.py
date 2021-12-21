import torch

class DualTensor(torch.Tensor):
    '''
    DualTensor objects store a primal tensor and its corresponding derivative
    tensor. They otherwise inherit all functionality from torch.Tensor,
    including size and requires_grad attributes.
    '''
    @staticmethod
    def __new__(cls, xd, x, *args, **kwargs):
        return super().__new__(cls, xd, *args, **kwargs)
    
    def __init__(self, xd, x):
        super().__init__()
        self.x = x

class Fwd2Rev(torch.nn.Module):
    '''
    A neural network layer whose only purpose is to glue a forward-mode AD and
    reverse-mode AD together. It will store forward-propagated derivatives
    during the forward sweep, and combine them with the reverse-propagated
    derivatives during the reverse sweep, immediately resulting in the gradient
    tensors for all preceding forward layers.
    '''
    class __Func__(torch.autograd.Function):
        @staticmethod
        def forward(ctx, fwdinput):
            input_p = torch.Tensor(fwdinput.x)
            input_d = torch.Tensor(fwdinput)
            ctx.save_for_backward(input_d)
            return input_p

        @staticmethod
        def backward(ctx, grad_output):
            input_d, = ctx.saved_tensors
            grad_input = input_d*grad_output
            return grad_input

    def __init__(self):
        super(Fwd2Rev, self).__init__()

    def forward(self, x):
        return self.__Func__.apply(x)

class Conv2d(torch.nn.Module):
    class __Func__(torch.autograd.Function):
        @staticmethod
        def forward(ctx, fwdinput, weight, bias=None):
            if(isinstance(fwdinput, list)):
                fwdinput, input_d = fwdinput
            with torch.no_grad():
                # Primal Evaluation
                ret = torch.nn.functional.conv2d(fwdinput, weight, bias)
                # Derivative Propagation
                n_directions = 0
                if ctx.needs_input_grad[0]:
                    return None, None, None
                if ctx.needs_input_grad[1]:
                    n_directions += 9
                if bias is not None and ctx.needs_input_grad[2]:
                    n_directions += 1
                bias_d = torch.nn.Parameter(torch.zeros(n_directions), requires_grad=False)
                weight_d = torch.nn.Parameter(torch.zeros(n_directions, 1, 3, 3), requires_grad=False)
                if ctx.needs_input_grad[1]:
                  for i in range(9):
                      weight_d[i,0,i//3,i%3] = 1
                if bias is not None and ctx.needs_input_grad[2]:
                  bias_d[-1] = 1
                ret_d = torch.nn.functional.conv2d(fwdinput, weight_d, bias_d)
            ret.requires_grad = True
            ret_dual = DualTensor(ret_d, ret)
            return ret_dual

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = None
            grad_weight = grad_output[:,0:9,:,:].sum(dim=[0,2,3]).reshape(1,1,3,3) if ctx.needs_input_grad[1] else None
            grad_bias = grad_output[:,-1,:,:].sum(dim=[0]) if ctx.needs_input_grad[2] else None
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
