import automad
import torch

n_batches = 1
n_chann = 1
n_out = 1

class Net_AutoMAD(torch.nn.Module):
    def __init__(self, mode="jacobian"):
        super(Net_AutoMAD, self).__init__()
        self.lin1 = torch.nn.Linear(2, 2)
        self.r2f = automad.Rev2Fwd()
        self.lin2 = automad.Linear(2, 1, mode=mode)
        self.f2r = automad.Fwd2Rev()
        self.mse = torch.nn.MSELoss()

    def forward(self, x, tgt=None):
        x = self.lin1(x)
        x = self.r2f(x)
        x = self.lin2(x)
        x = self.f2r(x)
        if(tgt != None):
            x = self.mse(x, tgt)
        return x


class Net_AutoGrad(torch.nn.Module):
    def __init__(self):
        super(Net_AutoGrad, self).__init__()
        self.lin1 = torch.nn.Linear(2, 2)
        self.lin2 = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        return x

nninput = torch.ones(n_batches, 2)
tgt = torch.zeros(n_batches, 1)

'''
in: [1, 1]
x1: [[1,2][3,4]] x [1,1] => [3,7]
x2: [5,7] x [3,7] => [15+49] => [64]
loss = (64-0)^2 => 4096
tgt: [0]

dloss/dx2 = 2*(64)
dx2/dw = ...
'''

def test_forward_reverse():
    netrev = Net_AutoGrad()
    netrev.zero_grad()
    netfwd = Net_AutoMAD(mode="jacobian")
    w1 = torch.Tensor([[1,2],[3,4]])
    b1 = torch.Tensor([0,0])
    w2 = torch.Tensor([[5,7]])
    b2 = torch.Tensor([0])
    netrev.lin1.weight = torch.nn.Parameter(w1)
    netrev.lin1.bias = torch.nn.Parameter(b1)
    netrev.lin2.weight = torch.nn.Parameter(w2)
    netrev.lin2.bias = torch.nn.Parameter(b2)
    netfwd.lin1.weight = torch.nn.Parameter(netrev.lin1.weight)
    netfwd.lin1.bias = torch.nn.Parameter(netrev.lin1.bias)
    netfwd.lin2.weight = torch.nn.Parameter(netrev.lin2.weight)
    netfwd.lin2.bias = torch.nn.Parameter(netrev.lin2.bias)

    ########################
    # Conventional backprop
    ########################
    outrev = netrev(nninput)
    print("outrev:",outrev)
    lossrev = torch.nn.MSELoss(reduction='sum')
    lrev = lossrev(outrev, tgt)
    lrev.backward()
    #all_rev = torch.cat([param.grad.flatten().clone() for param in netrev.parameters() if param.grad != None])
    all_rev = [param.grad.flatten().clone() for param in netrev.parameters() if param.grad != None]
    print("rev:", all_rev)

    ########################
    # Forward gradients
    ########################
    netfwd.zero_grad()
    outfwd = netfwd(nninput, tgt)
    outfwd.backward()
    #all_fwd = torch.cat([param.grad.flatten().clone() for param in netfwd.parameters() if param.grad != None])
    all_fwd = [param.grad.flatten().clone() for param in netfwd.parameters() if param.grad != None]
    print("fwd:", all_fwd)


    #all_rev_n = all_rev/torch.norm(all_rev)
    #all_fwd_n = all_fwd/torch.norm(all_fwd)

    #

    #angle1 = torch.arccos(torch.clip(torch.dot(all_rev_n, all_fwd_n), -1.0, 1.0))
    #angle1 = angle1*180/torch.pi

    #print("REV:")
    #print(all_rev)
    #print("FWD:")
    #print(all_fwd)

    #print(f"Angle with {all_rev.size(0)} params: fwd_grad {angle1}")

if __name__ == '__main__':
    test_forward_reverse()
