import torch.nn as nn
import torchvision


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Resnet18(nn.Module):
    def __init__(self, out_features : int):
        super(Resnet18, self).__init__()
        self.out_features = out_features

        self.encoder = torchvision.models.resnet18(pretrained=True)
        in_features = self.encoder.fc.in_features
        
        self.encoder.fc = Identity()
        self.linear = nn.Linear(in_features, self.out_features)


    def forward(self, x):
        h = self.linear(self.encoder(x))
        return h

class Resnet50(nn.Module):
    def __init__(self, out_features : int):
        super(Resnet50, self).__init__()
        self.out_features = out_features

        self.encoder = torchvision.models.resnet50(pretrained=True)
        in_features = self.encoder.fc.in_features
        
        self.encoder.fc = Identity()
        self.linear = nn.Linear(in_features, self.out_features)


    def forward(self, x):
        h = self.linear(self.encoder(x))
        return h
