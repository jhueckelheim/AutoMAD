import automad
import torch

class Net(torch.nn.module):
    '''
    Create a neural network wherein the first few layers are torch.nn layers,
    followed by the rev2fwd layer, and then the remaining layers would be from automad
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(4, 5, 3)
        self.rev2fwd = automad.Rev2Fwd()
        self.avg = automad.AvgPool2d(2)
        self.linear = automad.Linear(5 * 6 * 6, 7)

    def call(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.rev2fwd(x)
        x = self.avg(x)
        x = self.linear(x)
        return x

n_batches = 2
nninput = torch.randn(n_batches, 3, 16, 16)
tgt = torch.randn(n_batches, 7)

def test_reverse_forward():
    # Reverse
    netrev = Net()
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

if __name__ == '__main__':
    test_forward_reverse()
    print('Check for equivalency of weights and biases between forward and reverse mode AD')
