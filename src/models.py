import lightning as L
import torch
from torch import optim, nn

class NSModel(nn.Module):
    def __init__(self, in_features=256+3, out_features=3):
        super(NSModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features, 256)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.linear5 = nn.Linear(64, 32)
        self.relu5 = nn.ReLU()
        self.linear6 = nn.Linear(32, 16)
        self.relu6 = nn.ReLU()
        self.linear7 = nn.Linear(16, out_features)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.relu4(x)
        x = self.linear5(x)
        x = self.relu5(x)
        x = self.linear6(x)
        x = self.relu6(x)
        x = self.linear7(x)
        return x

class VesselGeomEmbedding(nn.Module):
    def __init__(self, in_features=8192*3, out_features=256):
        super(VesselGeomEmbedding, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features, 2048)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2048, 1024)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(1024, out_features)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

