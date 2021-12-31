import automad
import torch

class Net_AutoMAD(torch.nn.Module):
    def __init__(self):
        super(Net_AutoMAD, self).__init__()
        self.conv1 = automad.Conv2d(1, 1, 3)
        self.f2r = automad.Fwd2Rev()
        self.conv2 = torch.nn.Conv2d(1, 1, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.f2r(x)
        x = self.conv2(x)
        return x

class Net_AutoGrad(torch.nn.Module):
    def __init__(self):
        super(Net_AutoGrad, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 3)
        self.conv2 = torch.nn.Conv2d(1, 1, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

n_batches = 7
nninput = torch.randn(n_batches, 1, 16, 16)
tgt = torch.randn(n_batches, 1, 12, 12)
##################
# reverse mode AD
##################
netrev = Net_AutoGrad()
netrev.conv1.bias = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
netrev.conv2.bias = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
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
#ensure we're using the same parameters and inputs as netrev
netfwd.conv1.weight = netrev.conv1.weight
netfwd.conv2.weight = netrev.conv2.weight
netfwd.conv1.bias = netrev.conv1.bias
netfwd.conv2.bias = netrev.conv2.bias
netfwd.zero_grad()
outfwd = netfwd(nninput)
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
