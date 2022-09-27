import automad
import torch
import numpy as np

class Net_AutoMAD(torch.nn.Module):
    def __init__(self, mode="jacobian"):
        super(Net_AutoMAD, self).__init__()
        self.conv1 = automad.Conv2d(3, 4, 3, mode=mode)
        self.tanh = automad.Tanh()
        self.conv2 = automad.Conv2d(4, 5, 3, mode=mode)
        self.dropout = automad.Dropout()
        self.relu = automad.ReLU()
        self.avg = automad.AvgPool2d(2)
        self.max = automad.MaxPool2d(2)
        self.linear = automad.Linear(5*3*3, 7, mode=mode)
        self.f2r = automad.Fwd2Rev()


    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.conv2(x)
        torch.manual_seed(42)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.avg(x)
        x = self.max(x)
        x = automad.flatten(x, 1)
        x = self.linear(x)
        x = self.f2r(x)
        return x


class Net_AutoGrad(torch.nn.Module):
    def __init__(self):
        super(Net_AutoGrad, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3)
        self.tanh = torch.nn.Tanh()
        self.conv2 = torch.nn.Conv2d(4, 5, 3)
        self.dropout = torch.nn.Dropout()
        self.relu = torch.nn.ReLU()
        self.avg = torch.nn.AvgPool2d(2)
        self.max = torch.nn.MaxPool2d(2)
        self.linear = torch.nn.Linear(5*3*3, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.conv2(x)
        torch.manual_seed(42)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.avg(x)
        x = self.max(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

n_batches = 2
nninput = torch.randn(n_batches, 3, 16, 16)
tgt = torch.randn(n_batches, 7)
#tgt = torch.randn(n_batches, 5, 6, 6)

def test_forward_reverse():
    # Reverse
    netrev = Net_AutoGrad()
    # Forward / Mixed
    netfwd = Net_AutoMAD(mode="jacobian")
    # ensure we're using the same parameters and inputs as netrev
    netfwd.conv1.weight = torch.nn.Parameter(netrev.conv1.weight)
    netfwd.conv2.weight = torch.nn.Parameter(netrev.conv2.weight)
    netfwd.conv1.bias = torch.nn.Parameter(netrev.conv1.bias)
    netfwd.conv2.bias = torch.nn.Parameter(netrev.conv2.bias)
    netfwd.linear.weight = torch.nn.Parameter(netrev.linear.weight)
    netfwd.linear.bias = torch.nn.Parameter(netrev.linear.bias)
    netrev.zero_grad()
    netfwd.zero_grad()

    outrev = netrev(nninput)
    lossrev = torch.nn.MSELoss(reduction='sum')
    lrev = lossrev(outrev, tgt)
    lrev.backward()
    outfwd = netfwd(nninput)
    lossfwd = torch.nn.MSELoss(reduction='sum')
    lfwd = lossfwd(outfwd, tgt)
    lfwd.backward()

    print("""
    REVERSE AD
            """)
    print(f"d_conv1_weight:\n{netrev.conv1.weight.grad}")
    print(f"d_conv1_bias:\n{netrev.conv1.bias.grad}")
    #print(f"d_conv2_weight:\n{netrev.conv2.weight.grad}")
    #print(f"d_conv2_bias:\n{netrev.conv2.bias.grad}")
    print("""
    MIXED MODE AD
            """)
    print(f"d_conv1_weight:\n{netfwd.conv1.weight.grad}")
    print(f"d_conv1_bias:\n{netfwd.conv1.bias.grad}")
    print(f"diff:\n{netfwd.conv1.weight.grad-netrev.conv1.weight.grad}")
    print(f"isclose:\n{torch.all(torch.isclose(netfwd.conv1.weight.grad, netrev.conv1.weight.grad, rtol=1e-05, atol=1e-06))}")
    #print(f"d_conv2_weight:\n{netfwd.conv2.weight.grad}")
    #print(f"d_conv2_bias:\n{netfwd.conv2.bias.grad}")
    rtol = 1e-4
    atol = 1e-5
    assert torch.all(torch.isclose(netrev.conv1.weight.grad, netfwd.conv1.weight.grad, rtol=rtol, atol=atol))
    assert torch.all(torch.isclose(netrev.conv1.bias.grad, netfwd.conv1.bias.grad, rtol=rtol, atol=atol))
    assert torch.all(torch.isclose(netrev.conv2.weight.grad, netfwd.conv2.weight.grad, rtol=rtol, atol=atol))
    assert torch.all(torch.isclose(netrev.conv2.bias.grad, netfwd.conv2.bias.grad, rtol=rtol, atol=atol))
    assert torch.all(torch.isclose(netrev.linear.weight.grad, netfwd.linear.weight.grad, rtol=rtol, atol=atol))
    assert torch.all(torch.isclose(netrev.linear.bias.grad, netfwd.linear.bias.grad, rtol=rtol, atol=atol))

if __name__ == '__main__':
    test_forward_reverse()
    print('Check for equivalency of weights and biases between forward and reverse mode AD')
