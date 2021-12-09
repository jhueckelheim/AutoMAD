import torch

class Fwd2Rev(torch.nn.Module):
    class __Func__(torch.autograd.Function):
        @staticmethod
        def forward(ctx, fwdinput):
            if(isinstance(fwdinput, list)):
                fwdinput, input_d = fwdinput
            ctx.save_for_backward(input_d)
            return fwdinput

        @staticmethod
        def backward(ctx, grad_output):
            print("IEUWBHFILUWEBPIUGB")
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
            return [ret, ret_d]

        @staticmethod
        def backward(ctx, grad_output):
            print("IEUWBHFILUWEBPIUGB")
            grad_input = None
            grad_weight = grad_output[:,0:9,:,:].sum(dim=[0,2,3]).reshape(1,1,3,3) if ctx.needs_input_grad[1] else None
            grad_bias = grad_output[:,-1,:,:].sum(dim=[0]) if bias is not None and ctx.needs_input_grad[2] else None
            print(grad_weight)
            print(grad_bias)
            return grad_input, grad_weight, grad_bias

    def __init__(self, in_channels, out_channels, kernel_size, bias = True):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias)
        self.weight = self.conv.weight
        self.bias = self.conv.bias

    def forward(self, x):
        return self.__Func__.apply(x, self.weight, self.bias)
