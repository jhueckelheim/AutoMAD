import automad
import torch
import numpy as np
import sys

n_batches = 1000
n_chann = 10
n_out = 2
img_size = 5
lin_in_size = n_chann*((img_size-2)**2)

class Net_AutoMAD(torch.nn.Module):
    def __init__(self, mode="jacobian"):
        super(Net_AutoMAD, self).__init__()
        self.conv1 = automad.Conv2d(n_chann, n_chann, 3, mode=mode)
        self.linear = automad.Linear(lin_in_size, n_out, mode=mode)
        self.mse = automad.MSELoss()
        self.f2r = automad.Fwd2Rev()

    def forward(self, x, tgt=None):
        x = self.conv1(x)
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
    netfwd = Net_AutoMAD(mode="nonjacobian")
    netfwd.conv1.weight = torch.nn.Parameter(netrev.conv1.weight)
    netfwd.conv1.bias = torch.nn.Parameter(netrev.conv1.bias)
    netfwd.linear.weight = torch.nn.Parameter(netrev.linear.weight)
    netfwd.linear.bias = torch.nn.Parameter(netrev.linear.bias)

    ########################
    # Conventional backprop
    ########################
    outrev = netrev(nninput)
    lossrev = torch.nn.MSELoss(reduction='sum')
    lrev = lossrev(outrev, tgt)
    lrev.backward()
    all_rev = torch.cat([param.grad.flatten().clone() for param in netrev.parameters() if param.grad != None])

    ########################
    # Forward gradients
    ########################
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

    ########################
    # Compare with dot prod
    ########################
    all_fwd_2 = torch.zeros(n_runs, n_grads)
    v = torch.normal(size=all_fwd_2.size(), mean=0.0, std=1.0)
    for i in range(n_runs):
        all_fwd_2[i,:] = sum(all_rev * v[i,:]) * v[i,:]
    all_fwd_2 = torch.mean(all_fwd_2, dim=0)

    ########################
    # Compare with QR
    ########################
    all_fwd_3 = torch.zeros(n_runs, n_grads)
    v = torch.normal(size=all_fwd_3.size(), mean=0.0, std=1.0)
    VQ, VR = torch.linalg.qr(v.transpose(0,1), mode='reduced')
    VQ = -VQ.transpose(0,1)
    for i in range(n_runs):
        all_fwd_3[i,:] = sum(all_rev * VQ[i,:]) * VQ[i,:]
    all_fwd_3 = torch.mean(all_fwd_3, dim=0)

    all_rev_n = all_rev/torch.norm(all_rev)
    all_fwd_n = all_fwd/torch.norm(all_fwd)
    all_fw2_n = all_fwd_2/torch.norm(all_fwd_2)
    all_fw3_n = all_fwd_3/torch.norm(all_fwd_3)

    

    angle1 = torch.arccos(torch.clip(torch.dot(all_rev_n, all_fwd_n), -1.0, 1.0))
    angle1 = angle1*180/torch.pi

    angle2 = torch.arccos(torch.clip(torch.dot(all_rev_n, all_fw2_n), -1.0, 1.0))
    angle2 = angle2*180/torch.pi

    angle3 = torch.arccos(torch.clip(torch.dot(all_rev_n, all_fw3_n), -1.0, 1.0))
    angle3 = angle3*180/torch.pi

    print(f"Angle with {n_runs} runs and {all_rev.size(0)} params: fwd_grad {angle1}, check_grad {angle2}, QR_grad {angle3}")

if __name__ == '__main__':
    n_runs = 1000
    if(len(sys.argv) > 1):
        n_runs = int(sys.argv[1])
    test_forward_reverse(n_runs)
