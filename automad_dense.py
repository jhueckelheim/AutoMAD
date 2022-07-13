import torch

def channel2batch(x, out_channels=1):
    # reinterpret a [x,y*c,:,:] tensor as a [x*c,y,:,:] tensor
    return x.view(x.size(0)*x.size(1)//out_channels, out_channels, x.size(2), x.size(3))

def truebatch2outer(x, n_batch):
    # reinterpret a [x*n_batch,:,:,:] tensor as a [n_batch,x,:,:,:] tensor
    return x.view(n_batch, x.size(0)//n_batch, x.size(1), x.size(2), x.size(3))

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
            input_d = truebatch2outer(input_d, grad_output.size(0))
            grad_output = grad_output.unsqueeze(1)
            grad_input = (input_d*grad_output).sum(dim=[0,2,3,4])
            return grad_input.flatten()

    def __init__(self):
        super(Fwd2Rev, self).__init__()

    def forward(self, x):
        return self.__Func__.apply(x)

class Conv2d(torch.nn.Module):
    class __Func__(torch.autograd.Function):
        @staticmethod
        def forward(ctx, fwdinput_dual, weight, bias, args, kwargs):
            with torch.no_grad():
                ###############################################################
                # Forward-propagation of incoming derivatives: $w*\dot{x}$
                ###############################################################
                if ctx.needs_input_grad[0]:
                    # In this case, the incoming tensor is already a DualTensor,
                    # which means we are probably not the first layer in this
                    # network. We extract the primal and derivative inputs.
                    fwdinput = fwdinput_dual.x
                    fwdinput_d = fwdinput_dual.xd
                    ret_d0 = torch.nn.functional.conv2d(fwdinput_d, weight, None, *args, **kwargs)
                    n_dervs_incoming = fwdinput_dual.size(0)
                else:
                    # In this case, the incoming tensor is just a plain tensor,
                    # so we are probably the first (differentiated) layer. There
                    # is no incoming derivative for us to forward-propagate.
                    fwdinput = fwdinput_dual
                    ret_d0 = None
                    n_dervs_incoming = 0

                ###############################################################
                # Evaluate new derivatives created by this layer:
                # $\dot{w}*x$ and $\dot{b}$
                ###############################################################
                # Determine size and shape of derivative objects for weight_d
                # and bias_d (the "new" derivatives with respect to the
                # trainable parameters of the current layer). The number of
                # derivatives is essentially the sum of weight.numel() and
                # bias.numel().
                # We also store the sizes and numbers of derivatives in `ctx`,
                # which allows us to retrieve this information during the
                # backwards pass.
                n_dervs = 0
                ctx.derv_dims = {}
                if ctx.needs_input_grad[1]:
                    n_dervs_w = weight.numel()
                    n_dervs += n_dervs_w
                    ctx.derv_dims['weight'] = [n_dervs_w, weight.shape]
                if bias is not None and ctx.needs_input_grad[2]:
                    n_dervs_b = bias.numel()
                    n_dervs += n_dervs_b
                    ctx.derv_dims['bias'] = [n_dervs_b, bias.shape]
                ctx.derv_dims['layer'] = n_dervs
                # Seeding. Make sure that either weight_d or bias_d (not both)
                # has exactly one entry "1.0", and all other entries "0.0", for
                # each derivative input channel.
                out_channels = weight.size(0)
                weight_d = torch.nn.Parameter(torch.zeros(n_dervs*out_channels, *weight.shape[1:]), requires_grad=False)
                bias_d = torch.nn.Parameter(torch.zeros(n_dervs*out_channels), requires_grad=False)
                if ctx.needs_input_grad[1]:
                    weight_d[0:n_dervs_w*out_channels,:,:,:].flatten()[0::n_dervs_w+1] = 1
                if bias is not None and ctx.needs_input_grad[2]:
                    bias_d[-n_dervs_b*out_channels::n_dervs_b+1] = 1
                # Derivative Propagation
                ret_d1 = torch.nn.functional.conv2d(fwdinput, weight_d, bias_d, *args, **kwargs)
                ret_d1 = channel2batch(ret_d1, out_channels)
                if(ret_d0 != None):
                    n_batch = fwdinput.size(0)
                    ret_d0 = truebatch2outer(ret_d0, n_batch)
                    ret_d1 = truebatch2outer(ret_d1, n_batch)
                    ret_d = torch.cat([ret_d0, ret_d1], dim=1)
                    ret_d = ret_d.view(ret_d.size(0)*ret_d.size(1), *ret_d.shape[2:])
                else:
                    ret_d = ret_d1

                ###############################################################
                # Primal Evaluation: $w*x+b$
                ###############################################################
                ret = torch.nn.functional.conv2d(fwdinput, weight, bias, *args, **kwargs)

            ret_dual = DualTensor(torch.zeros(n_dervs+n_dervs_incoming), ret, ret_d)
            return ret_dual

        @staticmethod
        def backward(ctx, grad_output):
            # grad_input is the pass-through of gradient information for layers
            # that happened before us in the forward pass (and will happen after
            # this in the reverse pass).
            # The gradient tensor is organized such that the last directional
            # derivatives belong to the current layer. Everything except those
            # last directions is passed into grad_input.
            grad_input = grad_output[:-ctx.derv_dims['layer']]
            # the rest is gradient info for the current layer. The weights are
            # first, the bias is last.
            if(ctx.needs_input_grad[1]):
                grad_weight = grad_output[-ctx.derv_dims['layer']:-ctx.derv_dims['layer']+ctx.derv_dims['weight'][0]].view(ctx.derv_dims['weight'][1])
            else:
                grad_weight = None
            if(ctx.needs_input_grad[2]):
                grad_bias = grad_output[-ctx.derv_dims['bias'][0]:].view(ctx.derv_dims['bias'][1])
            else:
                grad_bias = None
            return grad_input, grad_weight, grad_bias, None, None

    def __init__(self, in_channels, out_channels, kernel_size, bias = True, *args, **kwargs):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # TODO we create a Conv2d layer here simply to piggy-back on its default initialization for weight and bias.
        # We should just find out whatever the correct default initialization is, and use that.
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias)
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return self.__Func__.apply(x, self.weight, self.bias, self.args, self.kwargs)

class Tanh(torch.nn.Module):
    class __Func__(torch.autograd.Function):
        @staticmethod
        def forward(ctx, fwdinput_dual):
            with torch.no_grad():
                if ctx.needs_input_grad[0]:
                    # In this case, the incoming tensor is already a DualTensor,
                    # which means we are probably not the first layer in this
                    # network. We extract the primal and derivative inputs.
                    fwdinput = fwdinput_dual.x
                    fwdinput_d = fwdinput_dual.xd
                    n_dervs_incoming = fwdinput_dual.size(0)
                else:
                    # In this case, the incoming tensor is just a plain tensor,
                    # so we are probably the first (differentiated) layer. There
                    # is no incoming derivative for us to forward-propagate.
                    fwdinput = fwdinput_dual
                    n_dervs_incoming = 0

                ###############################################################
                # Primal Evaluation: $ret = tanh(x)$
                ###############################################################
                ret = torch.tanh(fwdinput)

                ###############################################################
                # Forward-propagation of incoming derivatives:
                # $\dot{ret} = (1-tanh^2(x))*\dot{x}$
                ###############################################################
                if ctx.needs_input_grad[0]:
                    n_batch = fwdinput.size(0)
                    fwdinput_d = truebatch2outer(fwdinput_d, n_batch)
                    ret_p = ret.unsqueeze(1)
                    ret_d = (1 - ret_p*ret_p)*fwdinput_d
                    ret_d = ret_d.view(ret_d.size(0)*ret_d.size(1), *ret_d.shape[2:])
                else:
                    ret_d = None

            ret_dual = DualTensor(torch.zeros(n_dervs_incoming), ret, ret_d)
            return ret_dual

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        return self.__Func__.apply(x)

class AvgPool2d(torch.nn.Module):
    class __Func__(torch.autograd.Function):
        @staticmethod
        def forward(ctx, fwdinput_dual, kernel_size, args, kwargs):
            with torch.no_grad():
                if ctx.needs_input_grad[0]:
                    # In this case, the incoming tensor is already a DualTensor,
                    # which means we are probably not the first layer in this
                    # network. We extract the primal and derivative inputs.
                    fwdinput = fwdinput_dual.x
                    fwdinput_d = fwdinput_dual.xd
                    n_dervs_incoming = fwdinput_dual.size(0)
                else:
                    # In this case, the incoming tensor is just a plain tensor,
                    # so we are probably the first (differentiated) layer. There
                    # is no incoming derivative for us to forward-propagate.
                    fwdinput = fwdinput_dual
                    n_dervs_incoming = 0

                ###############################################################
                # Primal Evaluation
                ###############################################################
                ret = torch.nn.functional.avg_pool2d(fwdinput, kernel_size=kernel_size, *args, **kwargs)

                ###############################################################
                # Forward-propagation of incoming derivatives
                ###############################################################
                if ctx.needs_input_grad[0]:
                    ret_d = torch.nn.functional.avg_pool2d(fwdinput_d, kernel_size=kernel_size, *args, **kwargs)
                else:
                    ret_d = None

            ret_dual = DualTensor(torch.zeros(n_dervs_incoming), ret, ret_d)
            return ret_dual

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, None, None, None

    def __init__(self, kernel_size, *args, **kwargs):
        super(AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return self.__Func__.apply(x, self.kernel_size, self.args, self.kwargs)
