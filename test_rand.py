import automad
import torch
import numpy as np

class Net_AutoMAD(torch.nn.Module):
    def __init__(self, mode="jacobian"):
        super(Net_AutoMAD, self).__init__()
        self.conv1 = automad.Conv2d(2, 1, 3, mode=mode)
        self.tanh = automad.Tanh()
        self.linear = automad.Linear(4, 2, mode=mode)
        self.f2r = automad.Fwd2Rev()
        self.mse = automad.MSELoss()

    def forward(self, x, tgt=None):
        x = self.conv1(x)
        x = self.tanh(x)
        x = automad.flatten(x, 1)
        x = self.linear(x)
        if tgt != None:
            x = self.mse(x, tgt)
        x = self.f2r(x)
        return x


class Net_AutoGrad(torch.nn.Module):
    def __init__(self):
        super(Net_AutoGrad, self).__init__()
        self.conv1 = torch.nn.Conv2d(2, 1, 3)
        self.tanh = torch.nn.Tanh()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

n_batches = 1
nninput = torch.randn(n_batches, 2, 4, 4)
tgt = torch.randn(n_batches, 2)

def test_forward_reverse():
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
    print("""
    REVERSE AD
            """)
    print(f"d_conv1_weight: \n{netrev.conv1.weight.grad}")
    print(f"d_conv1_bias:   \n{netrev.conv1.bias.grad}")
    print(f"d_linear_weight:\n{netrev.linear.weight.grad}")
    print(f"d_linear_bias:  \n{netrev.linear.bias.grad}")

    n_runs = 10000
    gradlinear_b = torch.Tensor(n_runs, *netrev.linear.bias.shape)
    gradlinear_w = torch.Tensor(n_runs, *netrev.linear.weight.shape)
    gradconv1_b = torch.Tensor(n_runs, *netrev.conv1.bias.shape)
    gradconv1_w = torch.Tensor(n_runs, *netrev.conv1.weight.shape)
    for i in range(n_runs):
        netfwd.zero_grad()
        outfwd = netfwd(nninput, tgt)
        outfwd.backward(torch.ones(1))
        gradlinear_b[i,:] = netfwd.linear.bias.grad.clone()
        gradlinear_w[i,:] = netfwd.linear.weight.grad.clone()
        gradconv1_b[i,:] = netfwd.conv1.bias.grad.clone()
        gradconv1_w[i,:] = netfwd.conv1.weight.grad.clone()
    gradconv1_w_m = torch.mean(gradconv1_w, dim=0)
    gradconv1_b_m = torch.mean(gradconv1_b, dim=0)
    gradlinear_w_m = torch.mean(gradlinear_w, dim=0)
    gradlinear_b_m = torch.mean(gradlinear_b, dim=0)
    print("""
    MIXED MODE AD
            """)
    print(f"d_conv1_weight:\n{gradconv1_w_m}")
    print(f"d_conv1_bias:  \n{gradconv1_b_m}")
    print(f"d_linear_weight:\n{gradlinear_w_m}")
    print(f"d_linear_bias:  \n{gradlinear_b_m}")
    rtol = 1e-0
    atol = 1e-0

    all_rev = torch.cat((netrev.conv1.weight.grad.flatten(), netrev.conv1.bias.grad, netrev.linear.weight.grad.flatten(), netrev.linear.bias.grad))
    all_fwd = torch.cat((gradconv1_w_m.flatten(), gradconv1_b_m, gradlinear_w_m.flatten(), gradlinear_b_m))
    all_rev_n = all_rev/torch.norm(all_rev)
    all_fwd_n = all_fwd/torch.norm(all_fwd)

    angle = torch.arccos(torch.clip(torch.dot(all_rev_n, all_fwd_n), -1.0, 1.0))
    print(angle)
    assert angle < 0.3

if __name__ == '__main__':
    test_forward_reverse()
    print('Check for equivalency of weights and biases between forward and reverse mode AD')
