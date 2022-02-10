import torch

def channel2batch(x, out_channels=1):
    # reinterpret a [x,y,:,:] tensor as a [x*y,1,:,:] tensor
    return x.view(x.size(0)*x.size(1)//out_channels, out_channels, x.size(2), x.size(3))

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
            print(f"F2R store input_d({input_d.size()})")
            ctx.save_for_backward(input_d)
            print(f"F2R forward return input_p({input_p.size()})")
            return input_p

        @staticmethod
        def backward(ctx, grad_output):
            input_d, = ctx.saved_tensors
            print(f"mul input_d({input_d.size()}) * grad_output({grad_output.size()})")
            grad_input = (input_d*grad_output).sum(dim=[1,2,3])
            print(f"F2R backward return grad_input({grad_input.size()})")
            return grad_input.flatten()

    def __init__(self):
        super(Fwd2Rev, self).__init__()

    def forward(self, x):
        return self.__Func__.apply(x)

class Conv2d(torch.nn.Module):
    class __Func__(torch.autograd.Function):
        @staticmethod
        def forward(ctx, fwdinput_dual, weight, bias=None):
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
                    ret_d0 = torch.nn.functional.conv2d(fwdinput_d, weight, bias=None)
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
                torch.set_printoptions(profile="full")
                print(f"n bias dervs {bias_d.sum()} {n_dervs_b}")
                #print(weight_d)
                # Derivative Propagation
                ret_d1 = torch.nn.functional.conv2d(fwdinput, weight_d, bias_d)
                ret_d1 = channel2batch(ret_d1, out_channels)
                if(ret_d0 != None):
                    ret_d = torch.cat([ret_d0, ret_d1], dim=0)
                else:
                    ret_d = ret_d1

                ###############################################################
                # Primal Evaluation: $w*x+b$
                ###############################################################
                ret = torch.nn.functional.conv2d(fwdinput, weight, bias)

            ret_dual = DualTensor(torch.zeros(n_dervs+n_dervs_incoming), ret, ret_d)
            print(f"C2D forward return ret_dual({ret_dual.size()}, {ret.size()} {ret_d.size()})")
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
            print(f"C2D backward return grad_in({grad_input.size()}) grad_w({grad_weight.size()}) grad_bias({grad_bias.size()})")
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
