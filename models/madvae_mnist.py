import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .blocks import ConvBlock
from .blocks import DeConvBlock

class MADVAEMNIST(nn.Module):
    def __init__(
        self,
        image_size=28,
        image_channels=1,
        h_dim=4096,
        z_dim=128,
        num_classes=10):

        super().__init__()

        self.image_size = image_size
        self.image_channels = image_channels
        self.h_dim = h_dim
        self.z_dim = z_dim

        # module for encoder
        self.c1 = ConvBlock(self.image_channels, 64, 5, 1, 2)
        self.c2 = ConvBlock(64, 64, 4, 2, 3)
        self.c3 = ConvBlock(64, 128, 4, 2, 1)
        self.c4 = ConvBlock(128, 256, 4, 2, 1)
        self.e_module = nn.Sequential(self.c1, self.c2, self.c3, self.c4)
        self.mu =nn.Linear(self.h_dim, self.z_dim)
        self.sigma = nn.Linear(self.h_dim, self.z_dim)
        # module for image decoder
        self.linear = nn.Linear(self.z_dim, self.h_dim)
        self.d1 = DeConvBlock(256, 128, 4, 2, 1)
        self.d2 = DeConvBlock(128, 64, 4, 2, 1)
        self.d3 = DeConvBlock(64, 64, 4, 2, 3)
        self.d4 = nn.ConvTranspose2d(64, self.image_channels, 5, 1, 2, bias=False)
        self.img_module = nn.Sequential(self.d1, self.d2, self.d3, self.d4)

        self.fc1 = nn.Linear(z_dim, z_dim)
        self.fc2 = nn.Linear(z_dim, num_classes)

    # Encoder
    def encode(self, x):
        self.batch_size = x.size(0)
        x = self.e_module(x)
        x = x.view(self.batch_size, -1)
        mean = self.mu(x)
        var = self.sigma(x)
        distribution = Normal(mean, var)

        return distribution
    
    def classifier(self, x):
        dist = self.encode(x)
        z = dist.rsample()

        out = F.relu(self.fc1(z))
        out = self.fc2(out)

        return F.log_softmax(out, dim=1)

    # Decoder for image denoising
    def img_decode(self, z):
        self.batch_size = z.size(0)
        x = F.relu(self.linear(z))
        x = x.view(self.batch_size, 256, 4, 4)

        return torch.sigmoid(self.img_module(x))
    
    # Forward function
    def forward(self, x):
        dist = self.encode(x)
        if self.training == True:
            z = dist.rsample()
        else:
            z = dist.mean
        output = self.img_decode(z)
        
        return output, dist.mean, dist.stddev, z