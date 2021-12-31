import torch

def channel2batch(x):
    # reinterpret a [x,y,:,:] tensor as a [x*y,1,:,:] tensor
    return x.view(x.size(1)*x.size(0),1,x.size(2),x.size(3))

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
        self.batches = x.size(0)  # number of true batches (in original network)
        self.channels = x.size(1) # number of true channels
        self.img_x = x.size(2)    # image size, in x dimension
        self.img_y = x.size(3)    # image size, in y dimension
        self.nbdirs = xd.size(1) / x.size(1) # number of directional derivatives

    def derivativesAsBatches(self):
      '''
      Return a view in which the directional derivatives are merged into the
      batches. This can be useful if the same operation should be applied to all
      directional derivatives across all true batches. (Where a "true" batch is
      one that existed in the original network, to which we add additional
      batches that are actually directional derivatives.)
      '''
      return self.view(self.batches*self.nbdirs, self.channels, self.img_x, self.img_y)

    def unsqueezeDerivatives(self):
      '''
      Return a view in which the directional derivatives are a separate
      dimension, just inside the batch dimension.
      '''
      ret = self.unsqueeze(0)
      ret = ret.view(self.batches, self.nbdirs, self.channels, self.img_x, self.img_y)
      return ret

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
            input_d = input_d.unsqueeze(0).view(grad_output.size(0),input_d.size(0)//grad_output.size(0),input_d.size(1),input_d.size(2),input_d.size(3))
            print(f"mul input_d*grad_output, sizes {input_d.size()} {grad_output.size()}")
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
            if(isinstance(fwdinput_dual, DualTensor)):
                print("DUALTENSOR")
                fwdinput = fwdinput_dual.x
                fwdinput_d = fwdinput_dual.xd
                num_dervs = fwdinput_dual.size(0)
            else:
                print(f"SOMETHING ELSE: {type(fwdinput_dual)}")
                fwdinput = fwdinput_dual
                fwdinput_d = None
                num_dervs = 0
            n_batches = fwdinput.size(0)
            with torch.no_grad():
                if ctx.needs_input_grad[0]:
                    print(f"ctx cat fwdinput {fwdinput.size()}, fwdinput_d {fwdinput_d.size()}")
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
                  for i in range(9):
                      weight_d[i,0,i//3,i%3] = 1
                if bias is not None and ctx.needs_input_grad[2]:
                  bias_d[-1] = 1
                ret_d2 = torch.nn.functional.conv2d(fwdinput, weight_d, bias_d)
                print(f"ret_d cat ret_d1 {ret_d1.size()}, retd2 {ret_d2.size()}")
                ret_d = torch.cat([ret_d1, channel2batch(ret_d2)], dim=0)
            ret_p.requires_grad = True
            ret_dual = DualTensor(torch.zeros(10+num_dervs), ret_p, ret_d)
            print(f"returning DualTensor({type(ret_d)} {type(ret)}) with sizes {ret_d.size()} {ret_p.size()}")
            return ret_dual

        @staticmethod
        def backward(ctx, grad_output):
            print(f"conv backw receiving grad_out with {grad_output.size()}")
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
