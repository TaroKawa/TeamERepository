import torch
import torch.nn as nn
import torch.nn.functional as F

def g_conv_unit(input_features, output_features):
    conv_unit = nn.Sequential(nn.ConvTranspose2d(input_features, output_features, 4, stride=2, padding=1),
                	nn.BatchNorm2d(output_features),
                    nn.ReLU())
    return conv_unit

def d_conv_unit(input_features, output_features):
    conv_unit = nn.Sequential(nn.Conv2d(input_features, output_features, 6, stride=2, padding=2),
                    nn.BatchNorm2d(output_features),
                    nn.LeakyReLU(0.2))
    return conv_unit

class Discriminator(nn.Module):

    def __init__(self, num_classes, ndf=128):
        super(Discriminator, self).__init__()

        self.num_classes = num_classes

        self.conv1 = d_conv_unit(3 + self.num_classes, ndf)
        self.conv2 = d_conv_unit(ndf, ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(4)
        self.linear = nn.Linear(ndf * 8, 1)

    def forward(self, x, y):
        n = x.size()[0]
        x = x.float()
        y = y.repeat(1,64*64).view(y.size(0), self.num_classes, 64, 64).float()
        x = torch.cat((x, y), 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avg_pool(x)
        x = x.view(n, -1)
        x = self.linear(x)
        x = F.sigmoid(x)

        return x

class Generator(nn.Module):

    def __init__(self, num_classes, input_dim=100, ngf=128):
        super(Generator, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.ngf = ngf

        self.project = nn.Linear(input_dim + self.num_classes, 4*4 * ngf * 8, bias=False)
        self.batch_norm1d = nn.BatchNorm1d(4*4* ngf * 8)
        self.conv1 = g_conv_unit(ngf * 8, ngf * 4)
        self.conv2 = g_conv_unit(ngf * 4, ngf * 2)
        self.conv3 = g_conv_unit(ngf * 2, ngf)
        self.conv4 = nn.ConvTranspose2d(ngf, 3, 4, stride=2, padding=1)

    def forward(self, z, y):
        n = z.size()[0]
        z = z.float()
        y = y.float()
        z = torch.cat((z, y), 1)
        z = F.relu(self.batch_norm1d(self.project(z))) ###may be not self.project(x)
        z = z.view((n, self.ngf*8, 4, 4)) #changed self.ngf*4 to self.ngf*8
        z = self.conv1(z)
        z = self.conv2(z)
        z = self.conv3(z)
        z = self.conv4(z)
        z = F.tanh(z)
        return z
