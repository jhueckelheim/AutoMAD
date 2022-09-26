# AutoMAD
Mixed Mode Automatic Differentiation for AutoGrad.

## Why mixed mode?
Reverse mode aka back-propagation tends to work well for functions with few outputs and many parameters, but it also tends to use a lot of memory and is difficult to implement on some hardware.

Forward mode automatic differentiation tends to work well with few parameters, but can handle any number of outputs, does not require memory for storage of intermediate results, and can work well on novel hardware since it computes derivatives forward-only, without the need for data flow reversal.

Mixed mode automatic differentiation allows mixing and matching both modes within the same network, using forward mode for some layers and back-propagation for others. If used well, it may be more efficient than either mode on its own. It can also be used to obtain gradients of networks where backpropagation is not implemented or not possible for some layer(s). 

## How to use it?
AutoMAD is a prototype, and the API may change in the future. For now, this is how it is done. Assume we have a network like this:
```
class Net_AutoGrad(torch.nn.Module):
    def __init__(self):
        super(Net_AutoGrad, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3)
        self.tanh = torch.nn.Tanh()
        self.max = torch.nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.max(x)
        return x
```
If we wanted to use forward-mode automatic differentiation for the `Conv2d` and `Tanh` layer, but use conventional back-propagation for the `MaxPoll2d` layer, all we need to do is this:
```
class Net_AutoMAD(torch.nn.Module):
    def __init__(self):
        super(Net_AutoMAD, self).__init__()
        self.conv1 = automad.Conv2d(3, 4, 3)
        self.tanh = automad.Tanh()
        self.fwd2rev = automad.Fwd2Rev()
        self.max = torch.nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = fwd2rev(x)
        x = self.max(x)
        return x
```
Here, we swap the `torch.nn.*` layers with `automad.*` layers to switch from back-prop to forward mode, and insert the glue layer `Fwd2Rev` to manage the transition between modes.

## What happens under the hood?
The AutoMAD layers compute the full Jacobian by creating a cartesian-basis seed vector for each trainable parameter. Forward propagation is implemented in terms of existing torch functions, so hardware acceleration tends to work out of the box. The glue layer has to combine the computed Jacobian of the predecessor layers with the gradient of the successor layers to compute the gradient of the predecessor layers. The `backward()` function is used to store these gradients in the appropriate places in all layers.


## What works, what doesn't
Not all layer types are in AutoMAD, the list is still growing. Also, there is not yet a functional `Rev2Fwd` layer.

There is experimental support for a randomized version of the forward mode, which doesn't require computing the full Jacobian and is thus much cheaper. See for example this paper, https://arxiv.org/pdf/2202.08587 for the idea. Unlike previous work that we're ware of, AutoMAD can combine randomized forward mode with conventional back-propagation, which allows computing exact gradients for some layers and approximate gradients for others.
