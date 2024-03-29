import automad
import torch
import numpy as np
import sys

n_batches = 1
n_chann = 2
n_out = 2
img_size = 20
lin_in_size = n_chann*((img_size-4)**2)//(2*2*2*2)

class Net_AutoMAD(torch.nn.Module):
    def __init__(self, mode="jacobian"):
        super(Net_AutoMAD, self).__init__()
        self.conv1 = automad.Conv2d(n_chann, n_chann, 3, mode=mode)
        self.tanh = automad.Tanh()
        self.conv2 = automad.Conv2d(n_chann, n_chann, 3, mode=mode)
        self.dropout = automad.Dropout()
        self.relu = automad.ReLU()
        self.avg = automad.AvgPool2d(2)
        self.max = automad.MaxPool2d(2)
        self.linear = automad.Linear(lin_in_size, n_out, mode=mode)
        self.mse = automad.MSELoss()
        self.f2r = automad.Fwd2Rev()

    def forward(self, x, tgt=None):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.avg(x)
        x = self.max(x)
        x = automad.flatten(x, 1)
        x = self.linear(x)
        if tgt != None:
            x = self.mse(x, tgt)
        x = self.f2r(x)
        return x


class Net_AutoGrad(torch.nn.Module):
    def __init__(self):
        super(Net_AutoGrad, self).__init__()
        self.conv1 = torch.nn.Conv2d(n_chann, n_chann, 3)
        self.tanh = torch.nn.Tanh()
        self.conv2 = torch.nn.Conv2d(n_chann, n_chann, 3)
        self.dropout = torch.nn.Dropout()
        self.relu = torch.nn.ReLU()
        self.avg = torch.nn.AvgPool2d(2)
        self.max = torch.nn.MaxPool2d(2)
        self.linear = torch.nn.Linear(lin_in_size, n_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.avg(x)
        x = self.max(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

nninput = torch.rand(n_batches, n_chann, img_size, img_size)
tgt = torch.rand(n_batches, n_out)

def test_forward_reverse(n_runs):
    netrev = Net_AutoGrad()
    netrev.zero_grad()
    netfwd = Net_AutoMAD(mode="nonjacobian")
    netfwd.conv1.weight = torch.nn.Parameter(netrev.conv1.weight)
    netfwd.conv1.bias = torch.nn.Parameter(netrev.conv1.bias)
    netfwd.linear.weight = torch.nn.Parameter(netrev.linear.weight)
    netfwd.linear.bias = torch.nn.Parameter(netrev.linear.bias)

    outrev = netrev(nninput)
    lossrev = torch.nn.MSELoss(reduction='sum')
    lrev = lossrev(outrev, tgt)
    lrev.backward()
    all_rev = torch.cat([param.grad.flatten().clone() for param in netrev.parameters() if param.grad != None])

    all_fwd = None
    for i in range(n_runs):
        netfwd.zero_grad()
        outfwd = netfwd(nninput, tgt)
        outfwd.backward(torch.ones(1))
        if(all_fwd == None):
            n_grads = 0
            for param in netfwd.parameters():
                if(param.grad != None):
                    n_grads += param.grad.numel()
            all_fwd = torch.zeros(n_runs, n_grads)
        all_fwd[i,:] = torch.cat([param.grad.flatten().clone() for param in netfwd.parameters() if param.grad != None])
    all_fwd = torch.mean(all_fwd, dim=0)
    rtol = 1e-0
    atol = 1e-0

    all_rev_n = all_rev/torch.norm(all_rev)
    all_fwd_n = all_fwd/torch.norm(all_fwd)

    angle = torch.arccos(torch.clip(torch.dot(all_rev_n, all_fwd_n), -1.0, 1.0))
    angle = angle*180/torch.pi
    print("Angle between true and approximate gradient after %d runs with %d params: %f degrees"%(n_runs,all_rev.size(0),angle))
    #assert angle < 30

if __name__ == '__main__':
    n_runs = 1000
    if(len(sys.argv) > 1):
        n_runs = int(sys.argv[1])
    test_forward_reverse(n_runs)
