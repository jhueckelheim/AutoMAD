import torch

def channel2batch(x, out_channels=1):
    # reinterpret a [x,y*c,:,:] tensor as a [x*c,y,:,:] tensor
    return x.view(x.size(0)*x.size(1)//out_channels, out_channels, *x.shape[2:])

def truebatch2outer(x, n_batch):
    # reinterpret a [x*n_batch,:,:,:] tensor as a [n_batch,x,:,:,:] tensor
    return x.view(n_batch, x.size(0)//n_batch, *x.shape[1:])

def flatten(x, start_dim=0, end_dim=-1):
    if(start_dim < 1):
        raise ValueError("flatten with start dimension <1 is not supported by automad")
    ret = DualTensor(x, x.x, x.xd)
    ret.x = torch.flatten(ret.x, start_dim, end_dim)
    ret.xd = torch.flatten(ret.xd, start_dim, end_dim)
    return ret

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
            product = input_d*grad_output
            grad_input = product.sum(dim=[0]+list(range(2,product.dim())))
            #grad_input = torch.matmul(input_d, grad_output.transpose(0,1))
            return grad_input.flatten()

    def __init__(self):
        super(Fwd2Rev, self).__init__()

    def forward(self, x):
        return self.__Func__.apply(x)

class ForwardUtilities:
    @staticmethod
    def propagate(img, img_d, weight, weight_d, bias, bias_d, func, reshape_biasd, *args, **kwargs):
        n_batch = img.size(0)
        out_channels = weight.size(0)
    
        ###############################################################
        # Forward-propagation of weight/bias derivatives:
        # $\dot{w}*x$ and $\dot{b}$
        ###############################################################
        ret_d0 = None
        if(weight_d != None):
            ret_d0 = func(img, weight_d, bias_d, *args, **kwargs)
            ret_d0 = channel2batch(ret_d0, out_channels)
        elif(bias_d != None):
            ret_d0 = reshape_biasd(bias_d, img, weight, *args, **kwargs)
            ret_d0 = channel2batch(ret_d0.contiguous(), out_channels)
    
        ###############################################################
        # Forward-propagation of img derivatives: $w*\dot{x}$
        ###############################################################
        if img_d != None:
            # In this case, the incoming tensor is already a DualTensor,
            # which means we are probably not the first layer in this
            # network. We extract the primal and derivative inputs.
            ret_d1 = func(img_d, weight, None, *args, **kwargs)
            ret_d1 = func(img_d, weight, None, *args, **kwargs)
    
        ###############################################################
        # Assembly of derivative tensor
        ###############################################################
            if ret_d0 == None:
                ret_d = ret_d1
            else:
                ret_d = ret_d0 + ret_d1
        else:
            ret_d = ret_d0
        return ret_d

    @staticmethod
    def seed_cartesian(dims):
        n_dervs = dims.numel()
        seed = torch.nn.Parameter(
                   torch.zeros(n_dervs*dims[0], *dims[1:]),
                   requires_grad=False)
        seed.flatten()[0::n_dervs+1] = 1
        return n_dervs, seed

    @staticmethod
    def seed_randomized(dims):
        n_dervs = 1
        seed = torch.normal(size=dims, mean=0.0, std=1.0)
        #seed = torch.zeros(size=dims)
        #seed.flatten()[0] = 1
        #print("seed_randomized ", seed)
        return n_dervs, seed

class Linear(torch.nn.Module):
    class __Func__(torch.autograd.Function):
        def forward(ctx, fwdinput_dual, weight, bias, seedfunc, accmfunc, mode, args, kwargs):
            def reshape_biasd(bias_d, img, *args, **kwargs):
                bias_d_shape = bias_d.shape
                n_batch = img.size(0)
                return bias_d.unsqueeze(0).expand(n_batch, *bias_d_shape)

            with torch.no_grad():
                if ctx.needs_input_grad[0]:
                    fwdinput = fwdinput_dual.x
                    n_batch = fwdinput.size(0)
                    fwdinput_d = fwdinput_dual.xd
                    ret_d_in = ForwardUtilities.propagate(
                                   fwdinput, fwdinput_d, weight, None, bias, None,
                                   torch.nn.functional.linear, None,
                                   *args, **kwargs)
                    ret_d_in = truebatch2outer(ret_d_in, n_batch)
                else:
                    fwdinput = fwdinput_dual
                    n_batch = fwdinput.size(0)
                    ret_d_in = None
                out_channels = weight.size(0)

                ctx.mode = mode
                ctx.derv_dims = {}
                ctx.derv_dims['layer'] = 0
                if ctx.needs_input_grad[1]:
                    n_dervs_w, weight_d = seedfunc(weight.shape)
                    ctx.weight_d = weight_d
                    ctx.derv_dims['layer'] += n_dervs_w
                    ctx.derv_dims['weight'] = [n_dervs_w, weight.shape]
                    ret_d_w = ForwardUtilities.propagate(
                                   fwdinput, None, weight, weight_d, bias, None,
                                   torch.nn.functional.linear, None,
                                   *args, **kwargs)
                    ret_d_w = truebatch2outer(ret_d_w, n_batch)
                if bias is not None and ctx.needs_input_grad[2]:
                    n_dervs_b, bias_d = seedfunc(bias.shape)
                    ctx.bias_d = bias_d
                    ctx.derv_dims['layer'] += n_dervs_b
                    ctx.derv_dims['bias'] = [n_dervs_b, bias.shape]
                    ret_d_b = ForwardUtilities.propagate(
                                   fwdinput, None, weight, None, bias, bias_d,
                                   torch.nn.functional.linear, reshape_biasd,
                                   *args, **kwargs)
                    ret_d_b = truebatch2outer(ret_d_b, n_batch)

                if bias is not None and ctx.needs_input_grad[2]:
                    ret_d = accmfunc([i for i in [ret_d_in, ret_d_w, ret_d_b] if i != None])
                else:
                    ret_d = accmfunc([i for i in [ret_d_in, ret_d_w] if i != None])
                if(mode == "jacobian"):
                    total_dervs = ret_d.size(1)
                    ret_d = ret_d.view(ret_d.size(0)*ret_d.size(1), *ret_d.shape[2:])
                else:
                    total_dervs = ret_d.size(1) // weight.size(0)

                ###############################################################
                # Primal Evaluation: $w*x+b$
                ###############################################################
                ret = torch.nn.functional.linear(fwdinput, weight, bias, *args, **kwargs)

            ret_dual = DualTensor(torch.zeros(total_dervs), ret, ret_d)
            return ret_dual

        @staticmethod
        def backward(ctx, grad_output):
            if ctx.mode != "jacobian":
                grad_input = grad_output
                if(ctx.needs_input_grad[1]):
                    grad_weight = grad_input * ctx.weight_d
                else:
                    grad_weight = None
                if(ctx.needs_input_grad[2]):
                    grad_bias = grad_input * ctx.bias_d
                else:
                    grad_bias = None
            else:
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
            return grad_input, grad_weight, grad_bias, None, None, None, None, None

    def __init__(self, in_channels, out_channels, bias = True, mode="jacobian", *args, **kwargs):
        super(Linear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # TODO we create a Linear layer here simply to piggy-back on its default initialization for weight and bias.
        # We should just find out whatever the correct default initialization is, and use that.
        self.linear = torch.nn.Linear(in_channels, out_channels, bias)
        self.weight = self.linear.weight
        self.bias = self.linear.bias
        self.mode = mode
        if(mode == "jacobian"):
            self.seedfunc = ForwardUtilities.seed_cartesian
            self.accmfunc = lambda x: torch.cat(x, dim=1)
        else:
            self.seedfunc = ForwardUtilities.seed_randomized
            self.accmfunc = lambda x: torch.sum(torch.cat(x, dim=1), dim=1)
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return self.__Func__.apply(x, self.weight, self.bias, self.seedfunc, self.accmfunc, self.mode, self.args, self.kwargs)

class Conv2d(torch.nn.Module):
    class __Func__(torch.autograd.Function):
        @staticmethod
        def forward(ctx, fwdinput_dual, weight, bias, seedfunc, accmfunc, mode, args, kwargs):
            def reshape_biasd(bias_d, img, weight, stride=1, padding=0, dilation=1,
                                 *args, **kwargs):
                from math import floor
                h = floor(((img.size(2)+(2*padding)-(dilation*(weight.size(2)-1))-1)/stride)+1)
                w = floor(((img.size(3)+(2*padding)-(dilation*(weight.size(3)-1))-1)/stride)+1)
                n_batch = img.size(0)
                return bias_d.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(n_batch, -1, w, h)

            with torch.no_grad():
                if ctx.needs_input_grad[0]:
                    fwdinput = fwdinput_dual.x
                    n_batch = fwdinput.size(0)
                    fwdinput_d = fwdinput_dual.xd
                    ret_d_in = ForwardUtilities.propagate(
                                   fwdinput, fwdinput_d, weight, None, bias, None,
                                   torch.nn.functional.conv2d, None,
                                   *args, **kwargs)
                    ret_d_in = truebatch2outer(ret_d_in, n_batch)
                else:
                    fwdinput = fwdinput_dual
                    n_batch = fwdinput.size(0)
                    ret_d_in = None
                out_channels = weight.size(0)

                ctx.mode = mode
                ctx.derv_dims = {}
                ctx.derv_dims['layer'] = 0
                if ctx.needs_input_grad[1]:
                    n_dervs_w, weight_d = seedfunc(weight.shape)
                    ctx.weight_d = weight_d
                    ctx.derv_dims['layer'] += n_dervs_w
                    ctx.derv_dims['weight'] = [n_dervs_w, weight.shape]
                    ret_d_w = ForwardUtilities.propagate(
                                   fwdinput, None, weight, weight_d, bias, None,
                                   torch.nn.functional.conv2d, None,
                                   *args, **kwargs)
                    ret_d_w = truebatch2outer(ret_d_w, n_batch)
                if bias is not None and ctx.needs_input_grad[2]:
                    n_dervs_b, bias_d = seedfunc(bias.shape)
                    ctx.bias_d = bias_d
                    ctx.derv_dims['layer'] += n_dervs_b
                    ctx.derv_dims['bias'] = [n_dervs_b, bias.shape]
                    ret_d_b = ForwardUtilities.propagate(
                                   fwdinput, None, weight, None, bias, bias_d,
                                   torch.nn.functional.conv2d, reshape_biasd,
                                   *args, **kwargs)
                    ret_d_b = truebatch2outer(ret_d_b, n_batch)

                ret_d = accmfunc([i for i in [ret_d_in, ret_d_w, ret_d_b] if i != None])
                if(mode == "jacobian"):
                    total_dervs = ret_d.size(1)
                    ret_d = ret_d.view(ret_d.size(0)*ret_d.size(1), *ret_d.shape[2:])
                else:
                    total_dervs = ret_d.size(1) // bias.size(0)

                ###############################################################
                # Primal Evaluation: $w*x+b$
                ###############################################################
                ret = torch.nn.functional.conv2d(fwdinput, weight, bias, *args, **kwargs)

            ret_dual = DualTensor(torch.zeros(total_dervs), ret, ret_d)
            return ret_dual

        @staticmethod
        def backward(ctx, grad_output):
            if ctx.mode != "jacobian":
                grad_input = grad_output
                if(ctx.needs_input_grad[1]):
                    grad_weight = grad_input * ctx.weight_d
                else:
                    grad_weight = None
                if(ctx.needs_input_grad[2]):
                    grad_bias = grad_input * ctx.bias_d
                else:
                    grad_bias = None
            else:
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
            return grad_input, grad_weight, grad_bias, None, None, None, None, None

    def __init__(self, in_channels, out_channels, kernel_size, bias = True, mode="jacobian", *args, **kwargs):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # TODO we create a Conv2d layer here simply to piggy-back on its default initialization for weight and bias.
        # We should just find out whatever the correct default initialization is, and use that.
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias)
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        self.mode = mode
        if(mode == "jacobian"):
            self.seedfunc = ForwardUtilities.seed_cartesian
            self.accmfunc = lambda x: torch.cat(x, dim=1)
        else:
            self.seedfunc = ForwardUtilities.seed_randomized
            self.accmfunc = lambda x: torch.sum(torch.cat(x, dim=1), dim=1)
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return self.__Func__.apply(x, self.weight, self.bias, self.seedfunc, self.accmfunc, self.mode, self.args, self.kwargs)
	
class MSELoss(torch.nn.Module):
    class __Func__(torch.autograd.Function):
        @staticmethod
        def forward(ctx, fwdinput_dual, tgt):
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
                # Primal Evaluation: $ret = sum((x_i-tgt_i)^2)$
                ###############################################################
                ret = torch.Tensor(1)
                ret[0] = torch.nn.functional.mse_loss(fwdinput, tgt, reduction="sum")

                ###############################################################
                # Forward-propagation of incoming derivatives:
                # $\dot{ret} = 2*(x-tgt) * \dot{x}$
                ###############################################################
                if ctx.needs_input_grad[0]:
                    n_batch = fwdinput.size(0)
                    fwdinput_d = truebatch2outer(fwdinput_d, n_batch)
                    tgt_p = tgt.unsqueeze(1)
                    fwdinput_p = fwdinput.unsqueeze(1)
                    ret_d = torch.zeros(fwdinput_d.size(1))
                    for i in range(fwdinput_d.size(1)):
                        ret_d[i] = torch.sum(fwdinput_d[:,i,:].flatten()*((fwdinput_p - tgt_p).flatten())*2.0)
                    diff = (fwdinput_p - tgt_p).expand(fwdinput_d.size())
                    ret_d2 = torch.sum(fwdinput_d*diff*2.0, dim=[0,2])
                    #print("ret_d:", ret_d)
                    #print("ret_d2:", ret_d2)
                    #ret_d = fwdinput_d.flatten()*((fwdinput_p - tgt_p).expand(fwdinput_d.size()).flatten())*2.0
                    #ret_d = torch.sum(ret_d).flatten()
                else:
                    ret_d = None

            ret_dual = DualTensor(torch.zeros(n_dervs_incoming), ret, ret_d)
            return ret_dual

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, None

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, x, tgt):
        return self.__Func__.apply(x, tgt)

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
            ctx.save_for_backward(input_d)
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
            grad_input = (input_d * grad_output).sum(dim=[0, 2, 3, 4])
            return grad_input.flatten()

    def __init__(self):
        super(Fwd2Rev, self).__init__()

    def forward(self, x):
        return self.__Func__.apply(x)

class Dropout(torch.nn.Module):
    class __Func__(torch.autograd.Function):
        @staticmethod
        def forward(ctx, fwdinput_dual, p):
            with torch.no_grad():
                if ctx.needs_input_grad[0]:
                    fwdinput = fwdinput_dual.x
                    fwdinput_d = fwdinput_dual.xd
                    n_dervs_incoming = fwdinput_dual.size(0)
                else:
                    fwdinput = fwdinput_dual
                    n_dervs_incoming = 0
                #######################################
                # Primal Evaluation: $ret = dropout(x)$
                #######################################
                ret = torch.nn.functional.dropout(fwdinput, p=p)

                ##############################################
                # Forward-propagation of incoming derivatives:
                # $\dot{ret} = 1/(1-p) if chosen$
                # $\dot{ret} = 0       else$
                ##############################################
                if ctx.needs_input_grad[0]:
                    n_batch = fwdinput.size(0)
                    fwdinput_d = truebatch2outer(fwdinput_d, n_batch)
                    ret_d = torch.where(ret!=0, 1.0/(1.0-p), 0.0)
                    ret_d = ret_d.unsqueeze(1) * fwdinput_d
                    ret_d = ret_d.view(ret_d.size(0) * ret_d.size(1), *ret_d.shape[2:])
                else:
                    ret_d = None
            ret_dual = DualTensor(torch.zeros(n_dervs_incoming), ret, ret_d)
            return ret_dual

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, None

    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, x, p=None):
        if(p==None):
            p = self.p
        return self.__Func__.apply(x, p)

class ReLU(torch.nn.Module):
    class __Func__(torch.autograd.Function):
        @staticmethod
        def forward(ctx, fwdinput_dual):
            with torch.no_grad():
                if ctx.needs_input_grad[0]:
                    fwdinput = fwdinput_dual.x
                    fwdinput_d = fwdinput_dual.xd
                    n_dervs_incoming = fwdinput_dual.size(0)
                else:
                    fwdinput = fwdinput_dual
                    n_dervs_incoming = 0
                ####################################
                # Primal Evaluation: $ret = relu(x)$
                ####################################
                ret = torch.nn.functional.relu(fwdinput)

                ##############################################
                # Forward-propagation of incoming derivatives:
                # $\dot{ret} = 0 if x <= 0$
                # $\dot{ret} = 1 if x > 0$
                ##############################################
                if ctx.needs_input_grad[0]:
                    n_batch = fwdinput.size(0)
                    fwdinput_d = truebatch2outer(fwdinput_d, n_batch)
                    fwdinput_p = fwdinput.unsqueeze(1)
                    ret_d = torch.where(fwdinput_p > 0, fwdinput_d, 0.0)
                    ret_d = ret_d.view(ret_d.size(0) * ret_d.size(1), *ret_d.shape[2:])
                else:
                    ret_d = None
            ret_dual = DualTensor(torch.zeros(n_dervs_incoming), ret, ret_d)
            return ret_dual

        @staticmethod
        def backward(ctx, grad_output):
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
        def backward(ctx, revinput):
            input_p = revinput.x
            input_d = revinput.xd
            print(f"R2F store input_d({input_d.size()})")
            '''
            There is a save_for_forward function, which can be found here:
            https://pytorch.org/docs/stable/_modules/torch/autograd/function.html#FunctionCtx.save_for_backward
            This makes the pattern matching for implementing the inverse of Fwd2Rev easier.
            '''
            ctx.save_for_forward(input_d)
            print(f"R2F backward return input_p({input_p.size()})")
            return input_p

        @staticmethod
        def forward(ctx, grad_output):
            input_d, = ctx.saved_tensors
            input_d = truebatch2outer(input_d, grad_output.size(0))
            grad_output = grad_output.unsqueeze(1)
            print(f"mul input_d({input_d.size()}) * grad_output({grad_output.size()})")
            grad_input = (input_d * grad_output).sum(dim=[0, 2, 3, 4])
            print(f"R2F forward return grad_input({grad_input.size()})")
            return grad_input.flatten()

    def __init__(self):
        super(Rev2Fwd, self).__init__()

    def forward(self, x): #inference step, probably also want a backward that will take care of gradients
        return self.__Func__.apply(x)

class MaxPool2d(torch.nn.Module):
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
                ret, indices = torch.nn.functional.max_pool2d(fwdinput,
                                                     kernel_size=kernel_size,
                                                     *args, **kwargs,
                                                     return_indices=True)
                ###############################################################
                # Forward-propagation of incoming derivatives
                ###############################################################
                if ctx.needs_input_grad[0]:
                    '''
                    The derivative of the max pooling layer should be 1.0
                    for the element that got selected, and 0.0 everywhere else 
                    '''
                    n_batch = fwdinput.size(0)
                    n_dervs = fwdinput_d.size(0) // n_batch
                    fwdinput_d_flat = truebatch2outer(fwdinput_d, n_batch).flatten(start_dim=-2)
                    index_flat = truebatch2outer(indices,n_batch).flatten(start_dim=-2).repeat(1,n_dervs,1,1)
                    ret_d = fwdinput_d_flat.gather(dim=-1, index=index_flat).view(n_batch*n_dervs,*ret.shape[1:])

                else:
                    ret_d = None

            ret_dual = DualTensor(torch.zeros(n_dervs_incoming), ret, ret_d)
            return ret_dual

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, None, None, None

    def __init__(self, kernel_size, *args, **kwargs):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return self.__Func__.apply(x, self.kernel_size, self.args, self.kwargs)
