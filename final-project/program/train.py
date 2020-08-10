import torch
import torch.nn as nn
from Network import Network, MAX_WORD_COUNT, BATCH_SIZE
from LipDataset import LipDataset
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import time
from jiwer import wer

validation_split = 0.2
shuffle_dataset = True
random_seed= 42

network = Network()
dataset = LipDataset()

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, sampler=valid_sampler)

optimizer = optim.Adam([parameters for parameters in network.parameters() if parameters.requires_grad], lr = 0.1)

def tensor_to_sentence(t, d):
    return (" ".join([d[index] for index in t.argmax(dim=1).tolist() if d[index] != ""]))


def train(): # 1 epoch
    for batch in train_loader:
        videos, labels = batch
        preds = network(videos)
        for i in range (MAX_WORD_COUNT):
            p = preds[:,i,:]
            l = labels[:,i]
            loss = F.cross_entropy(p,l)
            print (loss.item())
            bwd_start = time.time()
            loss.backward(retain_graph=True)
            t_bwd = time.time() - bwd_start
            print (t_bwd)
        optimizer.step()
        optimizer.zero_grad()

def evaluate():
    for batch in valid_loader:
        videos, labels = batch
        preds = network(videos)
        for i in range (BATCH_SIZE):
            p = tensor_to_sentence(preds[i], dataset.index_to_word)
            l = " ".join([dataset.index_to_word[index] for index in labels[i].tolist() if index != 0])
            print (wer(p,l))

def save():
    torch.save(network, 'model.pkl')

#evaluate()
train()
#save()

