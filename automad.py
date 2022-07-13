import torch

def channel2batch(x):
    # reinterpret a [x,y,:,:] tensor as a [x*y,1,:,:] tensor
    return x.view(x.size(0)*x.size(1), 1, x.size(2), x.size(3))

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
            print(f"F2R store input_d({input_d.size()})")
            ctx.save_for_backward(input_d)
            print(f"F2R forward return input_p({input_p.size()})")
            return input_p

        @staticmethod
        def backward(ctx, grad_output):
            input_d, = ctx.saved_tensors
            n_true_batch = grad_output.size(0)
            input_d = truebatch2outer(input_d, n_true_batch)
            print(f"mul input_d({input_d.size()}) * grad_output({grad_output.unsqueeze(1).size()})")
            grad_input = (input_d*grad_output.unsqueeze(1)).sum(dim=[0,2,3,4])
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
                if(isinstance(fwdinput_dual, DualTensor)):
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
                    fwdinput_d = None
                    n_dervs_incoming = 0
                # Determine the "true" batch size. We are about to expand the
                # batch size with members that correspond to directional
                # derivatives. Those will be "within" the actual batch. For
                # example, if the primal function had a batch size of 3, with
                # two directional derivatives, the new batch will be of size 6
                # where the derivatives for the first batch member are a
                # contiguous block, followed by a block containing the
                # derivatives for the second batch member, etc.
                n_true_batch = fwdinput.size(0)
                if ctx.needs_input_grad[0]:
                    # If we have incoming derivatives to forward-propagate, we
                    # concatenate them with the primal input. This will allow us
                    # to pipe both of them together through a single call to
                    # Conv2d, to compute the primal output and one of the
                    # derivative terms simultaneously.
                    fwdinput_pd = torch.cat([fwdinput, fwdinput_d], dim=0)
                else:
                    fwdinput_pd = fwdinput
                # Primal Evaluation (includes forward-propagation of incoming
                # derivatives with respect to input, i.e.  weight*fwdinput_d)
                ret = torch.nn.functional.conv2d(fwdinput_pd, weight, bias)
                # Now we undo the concatenation that we did previously. If we
                # didn't actually concatenate (because we had no incoming
                # derivatives), the ret_d1 vector will be empty.
                ret_p = ret[0:n_true_batch,:,:,:]
                ret_d1 = ret[n_true_batch:,:,:,:]

                # Determine size and shape of derivative objects for weight_d
                # and bias_d (the "new" derivatives with respect to the
                # trainable parameters of the current layer). The number of
                # derivatives is essentially the sum of weight.numel() and
                # bias.numel(), but depends on whether weight and/or bias are
                # actually active. TODO this is not true due to
                # sparsity/redundancy
                # We also store the sizes and numbers of derivatives in `ctx`,
                # which allows us to retrieve this information during the
                # backwards pass.
                num_dervs = 0
                if ctx.needs_input_grad[1]:
                    num_dervs_w = weight.size(1)*weight.size(2)*weight.size(3)
                    num_dervs += num_dervs_w
                    ctx.weight_size = weight.size()
                if bias is not None and ctx.needs_input_grad[2]:
                    num_dervs += 1
                # Seeding. Make sure that either weight_d or bias_d (not both)
                # has exactly one entry "1.0", and all other entries "0.0", for
                # each derivative input channel.
                bias_d = torch.nn.Parameter(torch.zeros(num_dervs), requires_grad=False)
                weight_d = torch.nn.Parameter(torch.zeros(num_dervs, *weight.shape[1:]), requires_grad=False)
                if ctx.needs_input_grad[1]:
                    weight_d[0:num_dervs_w,:,:,:].flatten()[0::num_dervs_w] = 1
                if bias is not None and ctx.needs_input_grad[2]:
                    bias_d[-1] = 1
                # Derivative Propagation
                ret_d2 = channel2batch(torch.nn.functional.conv2d(fwdinput, weight_d, bias_d))
                ret_d2 = ret_d2.expand(ret_d2.size(0), weight.size(0), *ret_d2.shape[2:])
                ctx.num_dervs = num_dervs * weight.size(0)
                if(ret_d1.size(0) > 0):
                    ret_d = torch.cat([ret_d1, ret_d2], dim=0)
                else:
                    ret_d = ret_d2
            ret_dual = DualTensor(torch.zeros(ctx.num_dervs+n_dervs_incoming), ret_p, ret_d)
            print(f"C2D forward return ret_dual({ret_dual.size()}, {ret_p.size()} {ret_d.size()})")
            return ret_dual

        @staticmethod
        def backward(ctx, grad_output):
            # grad_input is the pass-through of gradient information for layers
            # that happened before us in the forward pass (and will happen after
            # this in the reverse pass).
            # The gradient tensor is organized such that the last directional
            # derivatives belong to the current layer. Everything except those
            # last directions is passed into grad_input.
            grad_input = grad_output[:-ctx.num_dervs]
            # the rest is gradient info for the current layer. The weights are
            # first, the bias is last.
            if(ctx.needs_input_grad[1]):
                grad_weight = grad_output[-ctx.num_dervs:-ctx.num_dervs+ctx.weight_numel].view(ctx.weight_size)
            else:
                grad_weight = None
            if(ctx.needs_input_grad[2]):
                grad_bias = grad_output[-ctx.bias_numel:].view(ctx.bias_numel)
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
