import torch
import torch.nn as nn
from program.Tcn import MultiscaleMultibranchTCN

class Network(nn.Module):

    def __init__(self):
        super().__init__()
        tcn_options = {'kernel_size': [3,5,7],'num_layers': 4}
        self.tcn = MultiscaleMultibranchTCN(input_size = 512, 
                num_channels=[15*len(tcn_options['kernel_size'])]*tcn_options['num_layers'], 
                tcn_options= tcn_options, 
                dropout=0.2, 
                relu_type = 'relu')

    def forward(self, x):
        return self.tcn(x)


net = Network()
# tensor = torch.rand([1,512,5,3,3])
ts = torch.stack([torch.rand([1,512,5,3,3]) for i in range (10)])
ps = []
for t in torch.unbind(ts):
    t = torch.reshape(t, [15,3,512])
    p = net(t)
    print (p.shape)
    ps.append(p)
p = torch.stack(ps)
# ts = torch.reshape(ts, [-1,15,3,512])
print (p.shape)