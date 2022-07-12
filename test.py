import automad_dense as automad
import torch


class Net_AutoMAD(torch.nn.Module):
    def __init__(self):
        super(Net_AutoMAD, self).__init__()
        self.conv1 = automad.Conv2d(3, 4, 3)
        #self.tanh = automad.Tanh()
        self.relu = automad.ReLU()
        self.conv2 = automad.Conv2d(4, 5, 3)
        self.avg = automad.AvgPool2d(2)
        self.f2r = automad.Fwd2Rev()

    def forward(self, x):
        x = self.conv1(x)
        #x = self.tanh(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.avg(x)
        x = self.f2r(x)
        return x


class Net_AutoGrad(torch.nn.Module):
    def __init__(self):
        super(Net_AutoGrad, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3)
        #self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(4, 5, 3)
        self.avg = torch.nn.AvgPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.tanh(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.avg(x)
        return x

n_batches = 2
nninput = torch.randn(n_batches, 3, 16, 16)
tgt = torch.randn(n_batches, 5, 6, 6)
print('nninput:')
print(nninput.size())
print(nninput)
##################
# reverse mode AD
##################
netrev = Net_AutoGrad()
# netrev.conv1.bias = torch.nn.Parameter(torch.zeros(4), requires_grad=True)
# netrev.conv2.bias = torch.nn.Parameter(torch.zeros(5), requires_grad=True)
netrev.zero_grad()
outrev = netrev(nninput)
lossrev = torch.nn.MSELoss(reduction='sum')
lrev = lossrev(outrev, tgt)
lrev.backward()
print("""
REVERSE AD
        """)
print(f"d_conv1_weight:\n{netrev.conv1.weight.grad}")
print(f"d_conv1_bias:\n{netrev.conv1.bias.grad}")
print(f"d_conv2_weight:\n{netrev.conv2.weight.grad}")
print(f"d_conv2_bias:\n{netrev.conv2.bias.grad}")

##################
# forward mode AD
##################
netfwd = Net_AutoMAD()
# ensure we're using the same parameters and inputs as netrev
netfwd.conv1.weight = netrev.conv1.weight
netfwd.conv2.weight = netrev.conv2.weight
netfwd.conv1.bias = netrev.conv1.bias
netfwd.conv2.bias = netrev.conv2.bias
netfwd.zero_grad()
outfwd = netfwd(nninput) # TRACE BACK ERROR STARTS HERE
lossfwd = torch.nn.MSELoss(reduction='sum')
lfwd = lossfwd(outfwd, tgt)
lfwd.backward()
print("""
MIXED MODE AD
        """)
print(f"d_conv1_weight:\n{netfwd.conv1.weight.grad}")
print(f"d_conv1_bias:\n{netfwd.conv1.bias.grad}")
print(f"d_conv2_weight:\n{netfwd.conv2.weight.grad}")
print(f"d_conv2_bias:\n{netfwd.conv2.bias.grad}")


def test_forward_reverse():
    # Reverse
    netrev = Net_AutoGrad()
    netrev.zero_grad()
    outrev = netrev(nninput)
    lossrev = torch.nn.MSELoss(reduction='sum')
    lrev = lossrev(outrev, tgt)
    lrev.backward()
    # Forward / Mixed?
    netfwd = Net_AutoMAD()
    # ensure we're using the same parameters and inputs as netrev
    netfwd.conv1.weight = netrev.conv1.weight
    netfwd.conv2.weight = netrev.conv2.weight
    netfwd.conv1.bias = netrev.conv1.bias
    netfwd.conv2.bias = netrev.conv2.bias
    netfwd.zero_grad()
    outfwd = netfwd(nninput)
    lossfwd = torch.nn.MSELoss(reduction='sum')
    lfwd = lossfwd(outfwd, tgt)
    lfwd.backward()

    assert torch.equal(netrev.conv1.weight.grad, netfwd.conv1.weight.grad) is True
    assert torch.equal(netrev.conv1.bias.grad, netfwd.conv1.bias.grad) is True
    assert torch.equal(netrev.conv2.weight.grad, netfwd.conv2.weight.grad) is True
    assert torch.equal(netrev.conv2.bias.grad, netfwd.conv2.bias.grad) is True
    '''
    Is the where condition correct even if some points are satisfied vs not.
    Check that both branches are relu are hitting.
    => inputs to relu should have some positive and some negative
        (inputs are from tanh layer)
    => outputs are either 0 or 1, at least one output of both
    If this doesn't work, could swap around tanh and relu, and use an image to check.
    '''

if __name__ == '__main__':
    test_forward_reverse()
    print('Check for equivalency of weights and biases between forward and reverse mode AD')