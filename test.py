import automad
import torch
import numpy as np

class Net_AutoMAD(torch.nn.Module):
    def __init__(self):
        super(Net_AutoMAD, self).__init__()
        self.conv1 = automad.Conv2d(3, 4, 3)
        #self.tanh = automad.Tanh()
        self.relu = automad.ReLU()
        self.conv2 = automad.Conv2d(4, 5, 3)
        #self.avg = automad.AvgPool2d(2)
        self.max = automad.MaxPool2d(2)
        self.linear = automad.Linear(5*6*6, 7)
        self.f2r = automad.Fwd2Rev()

    def forward(self, x):
        x = self.conv1(x)
        #x = self.tanh(x)
        x = self.relu(x)
        x = self.conv2(x)
        #x = self.avg(x)
        x = self.max(x)
        #print(x.shape)
        x = automad.flatten(x, 1)
        x = self.linear(x)
        x = self.f2r(x)
        return x


class Net_AutoGrad(torch.nn.Module):
    def __init__(self):
        super(Net_AutoGrad, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3)
        #self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(4, 5, 3)
        #self.avg = torch.nn.AvgPool2d(2)
        self.max = torch.nn.MaxPool2d(2)
        self.linear = torch.nn.Linear(5*6*6, 7)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.tanh(x)
        x = self.relu(x)
        x = self.conv2(x)
        #x = self.avg(x)
        x = self.max(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    return output

# pool of square window of size=3, stride=2
#m = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
# pool of non-square window
#m = torch.nn.MaxPool2d((3, 2), stride=(2, 1), return_indices=True)
#input = torch.randn(1, 2, 4, 4)
#output, i = m(input)
#print('Test for maxpool2d')
#print('output size:' + str(output.size()))
#print(output)
#print('i size:' + str(i.size()))
#print(i)
#print('Test for retrieving indices')
#output2 = retrieve_elements_from_indices(output, i)
#print(output2)
'''
https://discuss.pytorch.org/t/maxpool2d-indexing-order/8281
https://discuss.pytorch.org/t/pooling-using-idices-from-another-max-pooling/37209/4
https://discuss.pytorch.org/t/fill-value-to-matrix-based-on-index/34698
'''
#print('Test zero tensor filled based on indices')
#a = torch.zeros(output.size())
#print(a.size())
#print(a)
#print(a.size(0))
#a[torch.arange(a.size(0)).unsqueeze(1), 1] = 1
#print(a)

#print('Test to numpy unravel')
#indx = np.unravel_index(torch.argmax(output), output.shape())
#print(indx)
n_batches = 2
nninput = torch.randn(n_batches, 3, 16, 16)
tgt = torch.randn(n_batches, 7)

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

    #print("""
    #REVERSE AD
    #        """)
    #print(f"d_conv1_weight:\n{netrev.conv1.weight.grad}")
    #print(f"d_conv1_bias:\n{netrev.conv1.bias.grad}")
    #print(f"d_conv2_weight:\n{netrev.conv2.weight.grad}")
    #print(f"d_conv2_bias:\n{netrev.conv2.bias.grad}")
    #print("""
    #MIXED MODE AD
    #        """)
    #print(f"d_conv1_weight:\n{netfwd.conv1.weight.grad}")
    #print(f"d_conv1_bias:\n{netfwd.conv1.bias.grad}")
    #print(f"d_conv2_weight:\n{netfwd.conv2.weight.grad}")
    #print(f"d_conv2_bias:\n{netfwd.conv2.bias.grad}")
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
