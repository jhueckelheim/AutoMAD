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
            print(f"F2R store input_d({input_d.size()})")
            ctx.save_for_backward(input_d)
            print(f"F2R forward return input_p({input_p.size()})")
            return input_p

        @staticmethod
        def backward(ctx, grad_output):
            input_d, = ctx.saved_tensors
            input_d = truebatch2outer(input_d, grad_output.size(0))
            grad_output = grad_output.unsqueeze(1)
            print(f"mul input_d({input_d.size()}) * grad_output({grad_output.size()})")
            grad_input = (input_d*grad_output).sum(dim=[0,2,3,4])
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
                # Derivative Propagation
                ret_d1 = torch.nn.functional.conv2d(fwdinput, weight_d, bias_d)
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
        def forward(ctx, fwdinput_dual, kernel_size):
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
                ret = torch.nn.functional.avg_pool2d(fwdinput, kernel_size=kernel_size)

                ###############################################################
                # Forward-propagation of incoming derivatives
                ###############################################################
                if ctx.needs_input_grad[0]:
                    ret_d = torch.nn.functional.avg_pool2d(fwdinput_d, kernel_size=kernel_size)
                else:
                    ret_d = None

            ret_dual = DualTensor(torch.zeros(n_dervs_incoming), ret, ret_d)
            return ret_dual

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, None

    def __init__(self, kernel_size):
        super(AvgPool2d, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        return self.__Func__.apply(x, self.kernel_size)


class Rev2Fwd(torch.nn.Module):
    '''
    A neural network layer whose only purpose is to glue reverse-mode AD and
    forward-mode AD together. It will store forward-propagated derivatives
    during the forward sweep, and combine them with the reverse-propagated
    derivatives during the reverse sweep, immediately resulting in the gradient
    tensors for all preceding forward layers.
    '''

    class __Func__(torch.autograd.Function):
        @staticmethod
        def backward(ctx, revinput):  # fwdinput
            input_p = revinput.x  # fwdinput
            input_d = revinput.xd  # fwdinput
            print(f"R2F store input_d({input_d.size()})")
            ctx.save_for_backward(input_d)
            print(f"F2R forward return input_p({input_p.size()})")
            return input_p

        '''
        If in a layer after forward to rev glue layer, then OK
        Everything before glue layer is the interesting stuff
        Forward mode would have computed the gradient with respect to the output,
        but don't currently have the output.
        The output comes from both the back propagation and forward.
        Glue layer combines the partials from forward and reverse modes.
        Conv layer was created using pattern matching, but results were always wrong.
        More than 1 input batch, or additional input channels threw the conv layer off.

        Look at an intermediate variable:
        temp = sin(x)
        y = cos(temp)

        Forward mode AD on above. X_dot, temp_dot, y_dot. What do they mean?
        ydot = der of y wrt x
        tempdot = der of temp wrt to input (which is x)
        forward mode = der of variable wrt to input
        xbar = der of output wrt x 
        tempbar = der of output wrt to temp
        ybar = der of output wrt to y => 1
        xbar = der of output wrt to x => 1

        glue layer has both tempdot and tempbar.
        forward to rev: layers before glue layer have tempdot, but they want to have tempbar
        [not correct] rev to forward: layers before glue layer have tempbar, but they want to have tempdot
        special layer to develop that feeds the correct partials to get the reverse sweep started.
        imagine the normal implementation of back propagation.
        does forward pass
        at some point they are hit by reverse pass
        at that point the reverse point has some reverse propagated sensitivities 
        find a way to make it look as if you had done the reverse pass up until the glue layer, 
            but have the reverse partials in the right places
        find a way to kick start the process in the middle of the network
        all the info is there, just need the plumbing 
        how to trick pytorch into doing it
        the loss function does something similar, multi dimensional tensor thing in the forward mode,
        which is reduced to a scalar loss
        in reverse mode, the loss function is basically what kicks start the whole thing
        Make this rev2forward look like a loss function to pytorch, even though it really isn't,
        and make sure that custom loss function layer/thing feeds the partials into the right places
        '''

        @staticmethod
        def forward(ctx, grad_output):
            input_d, = ctx.saved_tensors
            input_d = truebatch2outer(input_d, grad_output.size(0))
            grad_output = grad_output.unsqueeze(1)
            print(f"mul input_d({input_d.size()}) * grad_output({grad_output.size()})")
            grad_input = (input_d * grad_output).sum(dim=[0, 2, 3, 4])
            print(f"F2R backward return grad_input({grad_input.size()})")
            return grad_input.flatten()

    def __init__(self):
        super(Fwd2Rev, self).__init__()

    def forward(self, x):
        return self.__Func__.apply(x)

class ReLU(torch.nn.Module):
    '''
    TODO: convert torch's ReLU activation function into AutoMAD
    Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#ReLU
    Applies the rectified linear unit function element-wise:
    :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`
    Args:
        inplace: can optionally do the operation in-place. Default: ``False``
    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    .. image:: ../scripts/activation_images/ReLU.png
    Examples::
        >>> m = torch.nn.ReLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
      An implementation of CReLU - https://arxiv.org/abs/1603.05201
        >>> m = torch.nn.ReLU()
        >>> input = torch.randn(2).unsqueeze(0)
        >>> output = torch.cat((m(input),m(-input)))
    '''

    class __Func__(torch.autograd.Function):
        @staticmethod
        def forward(ctx, fwdinput_dual):
            with torch.no_grad():
                # print('Min Value of ctx:' + str(torch.min(ctx)))
                # print('Max Value of ctx:' + str(torch.max(ctx)))
                if ctx.needs_input_grad[0]:
                    fwdinput = fwdinput_dual.x
                    fwdinput_d = fwdinput_dual.xd
                    n_dervs_incoming = fwdinput_dual.size(0)
                else:
                    fwdinput = fwdinput_dual
                    n_dervs_incoming = 0
                ####################################
                # Primal Evaluation: $ret = tanh(x)$
                ####################################
                print('Min Value of fwdinput_dual:' + str(torch.min(fwdinput_dual)))
                print('Max Value of fwdinput_dual:' + str(torch.max(fwdinput_dual)))
                ret = torch.nn.functional.relu(fwdinput)

                ##############################################
                # Forward-propagation of incoming derivatives:
                # $\dot{ret} = 0 if x < 0$
                # $\dot{ret} = 1 if x > 0$
                # $\dot{ret} = 0 if x = 0$ so that matrix is more sparse, see references
                # https://stats.stackexchange.com/questions/333394/what-is-the-derivative-of-the-relu-activation-function
                # https://www.quora.com/How-do-we-compute-the-gradient-of-a-ReLU-for-backpropagation
                ##############################################
                if ctx.needs_input_grad[0]:
                    n_batch = fwdinput.size(0)
                    fwdinput_d = truebatch2outer(fwdinput_d, n_batch)
                    ret_p = ret.unsqueeze(1)
                    # print('fwdinput_d:')
                    # print(fwdinput_d.size())
                    # print('ret_p:')
                    # print(ret_p.size())
                    # ret_p: torch.Size([2, 1, 4, 14, 14]).
                    # Check if any of the values are greater than zero, or all?
                    # print(torch.all(fwdinput_d > 0))
                    ret_d = torch.where(fwdinput_d > 0, 1.0, 0.0)
                    print('ret_d size:')
                    print(ret_d.size())
                    '''
                    if torch.all(fwdinput_d > 0):
                        print('Went into first boolean check')
                        ret_d = torch.full(ret_p.size(), 1.0) # should have used fwdinput_d.size()
                    else:
                        print('Went into second boolean check')
                        print('ret_p size:')
                        print(ret_p.size())
                        ret_d = torch.full(ret_p.size(), 0.0)
                    '''

                    ret_d = ret_d.view(ret_d.size(0) * ret_d.size(1), *ret_d.shape[2:])
                else:
                    ret_d = None
            ret_dual = DualTensor(torch.zeros(n_dervs_incoming), ret, ret_d)
            print('Min Value of ret_dual:' + str(torch.min(ret_dual)))
            print('Max Value of ret_dual:' + str(torch.max(ret_dual)))
            return ret_dual

        @staticmethod
        def backward(ctx, grad_output):
            print('grad_output from backward:')
            print(grad_output.size())
            return grad_output

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x, ):
        return self.__Func__.apply(x)

class Rev2Fwd(torch.nn.Module):
    '''
    A neural network layer whose only purpose is to glue reverse-mode AD and
    forward-mode AD together. It will store forward-propagated derivatives
    during the forward sweep, and combine them with the reverse-propagated
    derivatives during the reverse sweep, immediately resulting in the gradient
    tensors for all preceding forward layers.
    '''

    class __Func__(torch.autograd.Function):
        @staticmethod
        def backward(ctx, revinput):  # fwdinput
            input_p = revinput.x  # fwdinput
            input_d = revinput.xd  # fwdinput
            print(f"R2F store input_d({input_d.size()})")
            ctx.save_for_backward(input_d)
            print(f"F2R forward return input_p({input_p.size()})")
            return input_p

        '''
        If in a layer after forward to rev glue layer, then OK
        Everything before glue layer is the interesting stuff
        Forward mode would have computed the gradient with respect to the output,
        but don't currently have the output.
        The output comes from both the back propagation and forward.
        Glue layer combines the partials from forward and reverse modes.
        Conv layer was created using pattern matching, but results were always wrong.
        More than 1 input batch, or additional input channels threw the conv layer off.

        Look at an intermediate variable:
        temp = sin(x)
        y = cos(temp)

        Forward mode AD on above. X_dot, temp_dot, y_dot. What do they mean?
        ydot = der of y wrt x
        tempdot = der of temp wrt to input (which is x)
        forward mode = der of variable wrt to input
        xbar = der of output wrt x 
        tempbar = der of output wrt to temp
        ybar = der of output wrt to y => 1
        xbar = der of output wrt to x => 1

        glue layer has both tempdot and tempbar.
        forward to rev: layers before glue layer have tempdot, but they want to have tempbar
        [not correct] rev to forward: layers before glue layer have tempbar, but they want to have tempdot
        special layer to develop that feeds the correct partials to get the reverse sweep started.
        imagine the normal implementation of back propagation.
        does forward pass
        at some point they are hit by reverse pass
        at that point the reverse point has some reverse propagated sensitivities 
        find a way to make it look as if you had done the reverse pass up until the glue layer, 
            but have the reverse partials in the right places
        find a way to kick start the process in the middle of the network
        all the info is there, just need the plumbing 
        how to trick pytorch into doing it
        the loss function does something similar, multi dimensional tensor thing in the forward mode,
        which is reduced to a scalar loss
        in reverse mode, the loss function is basically what kicks start the whole thing
        Make this rev2forward look like a loss function to pytorch, even though it really isn't,
        and make sure that custom loss function layer/thing feeds the partials into the right places
        '''

        @staticmethod
        def forward(ctx, grad_output):
            input_d, = ctx.saved_tensors
            input_d = truebatch2outer(input_d, grad_output.size(0))
            grad_output = grad_output.unsqueeze(1)
            print(f"mul input_d({input_d.size()}) * grad_output({grad_output.size()})")
            grad_input = (input_d * grad_output).sum(dim=[0, 2, 3, 4])
            print(f"F2R backward return grad_input({grad_input.size()})")
            return grad_input.flatten()

    def __init__(self):
        super(Fwd2Rev, self).__init__()

    def forward(self, x):
        return self.__Func__.apply(x)