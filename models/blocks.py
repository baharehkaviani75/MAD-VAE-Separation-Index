import torch
import torch.nn as nn
import torch.nn.functional as F

# Class for convolution block
class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=0, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x
    
# Class for de-convolution block
class DeConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=0, stride=1, padding=0, out_padding=0):
        super(DeConvBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, out_padding, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x

# Class for residual conv block
class ResidualConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=0, stride=1, padding=0):
        super(ResidualConvBlock, self).__init__()
        self.pad1 = nn.ReplicationPad2d(padding)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, stride)
        self.norm1 = nn.InstanceNorm2d(dim)
        self.relu1 = nn.ReLU(True)
        self.pad2 = nn.ReplicationPad2d(padding)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, stride)
        self.norm2 = nn.InstanceNorm2d(dim)
        self.relu2 = nn.ReLU(True)
        self.module = nn.Sequential(self.pad1, self.conv1, self.norm1, self.relu1, self.pad2, self.conv2, self.norm2, self.relu2)
    
    def forward(self, x):
        return x + self.module(x)
    

class Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = Block(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                Block(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out