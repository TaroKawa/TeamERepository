import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

# random generator
def generate_z(batch_size):
    return torch.randn((2, 100))

# define the Generator
def g_conv_unit(input_features, output_features):
    conv_unit = nn.Sequential(nn.ConvTranspose2d(input_features, output_features, 4, stride=2, padding=1),
                    nn.BatchNorm2d(output_features),
                    nn.ReLU())
    return conv_unit

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
        x = torch.cat((z, y), 1)
        x = F.relu(self.batch_norm1d(self.project(x)))
        x = x.view((n, self.ngf*8, 4, 4))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = F.tanh(x)

        return x

# generate the image
def generate_save_image(category, output_dir):

    net = Generator(14).cuda()
    #th = torch.load('static/weight.pt')
    #net.load_state_dict(th)

    y = np.eye(14)[np.array([category, 0])]
    y = torch.Tensor(y).float().cuda()
    y = Variable(y)

    z = generate_z(2).cuda()
    z = Variable(z)

    img = net(z, y).data.cpu()
    img = (img + 1) * 127.5
    img = img[0].numpy().astype(np.uint8).transpose(1, 2, 0)
    img = Image.fromarray(img)

    img.save(os.path.join(output_dir, 'generated_image.png'), 'PNG')
