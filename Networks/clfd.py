import torch.nn as nn
import torchvision

from tcn import TCN


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class CLfD(nn.Module):
    def __init__(self, backbone:str, out_features : int, projection_dim : int):
        super(CLfD, self).__init__()

        self.out_features = out_features
        if backbone =="resnet18": 
            self.encoder = torchvision.models.resnet18(pretrained=True)
            in_features = self.encoder.fc.in_features
            
            self.encoder.fc = Identity()
            self.linear = nn.Linear(in_features, self.out_features)
        elif backbone == "resnet50":
            self.encoder = torchvision.models.resnet50(pretrained=True)
            in_features = self.encoder.fc.in_features
            
            self.encoder.fc = Identity()
            self.linear = nn.Linear(in_features, self.out_features)
        elif backbone == "tcn":
            self.encoder = TCN(out_features = self.out_features)
            self.linear = Identity()
        else:
            raise NotImplementedError

        self.projector = nn.Sequential(
            nn.Linear(self.out_features, self.out_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.out_features, projection_dim, bias=False),
        )

    def forward(self, x_a, x_p):
        h_a = self.linear(self.encoder(x_a))
        h_p = self.linear(self.encoder(x_p))

        z_a = self.projector(h_a)
        z_p = self.projector(h_p)
        return h_a, h_a, z_p, z_p
