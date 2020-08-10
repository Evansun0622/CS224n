import torch
import torch.nn as nn
from LipDataset import LipDataset
import torch.nn.functional as F
import Resnet
import torch.optim as optim
from Tcn import MultiscaleMultibranchTCN
MAX_WORD_COUNT = 24
VOCAB = 17978
BATCH_SIZE = 2

class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=6, kernel_size=5, groups=1, stride = 1)
        self.conv2 = nn.Conv3d(in_channels=6, out_channels=18, kernel_size=5, groups=1, stride = 1)

        self.resnet = Resnet.r3d_18()

        tcn_options = {'kernel_size': [3,5,7],'num_layers': 3}
        self.tcn = MultiscaleMultibranchTCN(input_size = 512, 
                num_classes = 500,
                num_channels=[15*len(tcn_options['kernel_size'])]*tcn_options['num_layers'], 
                tcn_options= tcn_options, 
                dropout=0.2, 
                relu_type = 'relu')

        self.lin1 = nn.Linear(2025,VOCAB*MAX_WORD_COUNT)
        # self.m = nn.Softmax(dim=1)

        # self.lstm = nn.GRU(135, 32, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True)


        

    def forward(self,t):
        tm = time.time() 
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool3d(t, kernel_size = 2, stride = 2)
        
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool3d(t, kernel_size = 2, stride = 2)

        t = self.resnet(t)
        if t.shape[0] != 1: #input is a batch
            all_t = torch.unbind(t)
            t = torch.stack([self.tcn(torch.reshape(t0,[15,3,512])) for t0 in all_t])
        else:
            t = self.tcn(torch.reshape(t,[15,3,512]))
            t = torch.unsqueeze(t,0)
        t = torch.flatten(t, start_dim = 1)

        t = self.lin1(t)

        t = t.reshape(-1,MAX_WORD_COUNT,VOCAB)

        # t = torch.reshape(t,[-1,15,45*3])
        # t = self.lstm(t)
        return t

if __name__ == "__main__":
    #for testing purposes
    network = Network()
    train_set = LipDataset()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam([parameters for parameters in network.parameters() if parameters.requires_grad], lr = 0.1)
    for batch in train_loader:
        videos, labels = batch
        preds = network(videos)
        for i in range (MAX_WORD_COUNT):
            p = preds[:,i,:].clone()
            l = labels[:,i].clone()
            loss = F.cross_entropy(p,l)
            print (loss.item())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
    
