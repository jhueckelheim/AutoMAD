import automad
import torch
import numpy as np
import sys

n_batches = 1
n_chann = 1
n_out = 2
img_size = 7
lin_in_size = n_chann*((img_size-2)**2)

class Net_AutoMAD(torch.nn.Module):
    def __init__(self, mode="jacobian"):
        super(Net_AutoMAD, self).__init__()
        self.conv1 = automad.Conv2d(n_chann, n_chann, 3, mode=mode)
        self.linear = automad.Linear(lin_in_size, n_out, mode=mode)
        self.mse = automad.MSELoss()
        self.f2r = automad.Fwd2Rev()

    def forward(self, x, tgt=None):
        xd = {}
        x = self.conv1(x)
        xd['conv1'] = x.xd
        print("conv1 out:", x.x.size())
        print("conv1 out_d:", xd['conv1'].size())
        print("conv1 params:", sum([p.numel() for p in self.conv1.parameters()])//2)
        x = torch.flatten(x.x, 1)
        x.requires_grad = True
        x = self.linear(x)
        xd['linear'] = x.xd
        print("lin out:", x.x.size())
        print("lin out_d:", xd['linear'].size())
        print("lin params:", sum([p.numel() for p in self.linear.parameters()])//2)
        return x.x, xd


class Net_AutoGrad(torch.nn.Module):
    def __init__(self):
        super(Net_AutoGrad, self).__init__()
        self.conv1 = torch.nn.Conv2d(n_chann, n_chann, 3)
        self.linear = torch.nn.Linear(lin_in_size, n_out)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

nninput = torch.rand(n_batches, n_chann, img_size, img_size)
tgt = torch.rand(n_batches, n_out)

def test_forward_reverse(n_runs):
    netrev = Net_AutoGrad()
    netrev.zero_grad()
    netfwd = Net_AutoMAD(mode="layer-jacobian")
    netfwd.conv1.weight = torch.nn.Parameter(netrev.conv1.weight)
    netfwd.conv1.bias = torch.nn.Parameter(netrev.conv1.bias)
    netfwd.linear.weight = torch.nn.Parameter(netrev.linear.weight)
    netfwd.linear.bias = torch.nn.Parameter(netrev.linear.bias)

    outrev = netrev(nninput)
    lossrev = torch.nn.MSELoss(reduction='sum')
    lrev = lossrev(outrev, tgt)
    lrev.backward()
    all_rev = torch.cat([param.grad.flatten().clone() for param in netrev.parameters() if param.grad != None])

    netfwd.zero_grad()
    outfwd = netfwd(nninput, tgt)
    print(outfwd[0].size(), outfwd[1]['conv1'].size(), outfwd[1]['linear'].size())

if __name__ == '__main__':
    n_runs = 1000
    if(len(sys.argv) > 1):
        n_runs = int(sys.argv[1])
    test_forward_reverse(n_runs)
