from __future__ import print_function
import argparse
import os
import numpy as np

import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='VAE Fashion MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--data_dir', type=str, default='data',
                    help='path of data directory')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--nf', type=int, default=32, help='number of features')
parser.add_argument('--result', type=str, default='results', help='dir of results')
args = parser.parse_args()

# prepare dataset
trans = transforms.Compose([
       transforms.Resize((32, 32)),
       transforms.ToTensor()
])
class TargetTrans():
    def __self__(self):
        pass

    def __call__(self, y):
        return np.eye(10)[y]

train_dataset = datasets.FashionMNIST(root = args.data_dir, download=True, transform=trans, target_transform=TargetTrans())
train_loader = data_utils.DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=1)
test_dataset = datasets.FashionMNIST(root = args.data_dir, download=True, transform = trans, target_transform=TargetTrans(), train=False)
test_loader = data_utils.DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=1)

# define the network
def decode_unit(input_features, output_features):
    conv_unit = nn.Sequential(nn.ConvTranspose2d(input_features, output_features, 4, stride=2, padding=1),
                    nn.BatchNorm2d(output_features),
                    nn.LeakyReLU(0.2))
    return conv_unit

def encode_unit(input_features, output_features):
    conv_unit = nn.Sequential(nn.Conv2d(input_features, output_features, 6, stride=2, padding=2),
                    nn.BatchNorm2d(output_features),
                    nn.LeakyReLU(0.2))
    return conv_unit

class CVAE(nn.Module):
    def __init__(self, nf=32):
        super(CVAE, self).__init__()
        self.nf = nf

        self.conv1 = encode_unit(11, nf)
        self.conv2 = encode_unit(nf, 2 * nf)
        self.conv3 = encode_unit(2 * nf, 4 * nf) 
        self.conv4 = encode_unit(4 * nf, 8 * nf)
        self.fc1 = nn.Linear(4*8*nf, 100)
        self.fc2 = nn.Linear(4*8*nf, 100)

        self.project = nn.Linear(110, 4*8*nf , bias=False)
        self.batch_norm1d = nn.BatchNorm1d(4*8*nf)
        self.dconv4 = decode_unit(8 * nf, 4 * nf)
        self.dconv3 = decode_unit(4 * nf, 2 * nf)
        self.dconv2 = decode_unit(2 * nf, nf)
        self.dconv1 = decode_unit(nf, 1)

    def encode(self, x, y):
        y = y.repeat(1,1024).view(y.size(0), 10, 32, 32)
        x = torch.cat((x, y), 1)
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = h.view(h.size(0), -1)
        return self.fc1(h), self.fc2(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = Variable(torch.randn(std.size()).cuda())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, y):
        z = torch.cat((z, y), 1)
        h = self.batch_norm1d(self.project(z))
        h = F.leaky_relu(h, negative_slope=0.2, inplace=True)
        h = h.view(h.size(0), 8 * self.nf, 2, 2) 
        h = self.dconv4(h)
        h = self.dconv3(h)
        h = self.dconv2(h)
        h = self.dconv1(h)
        return F.sigmoid(h)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar


model = CVAE(nf = args.nf).cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)

    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, y) in enumerate(train_loader):
        data = data.cuda()
        data = Variable(data)
        y = Variable(y.float().cuda())
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, y)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data.cpu()[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data.cpu()[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    #with torch.no_grad():
    for i, (data, y) in enumerate(test_loader):
        data = Variable(data.cuda())
        y = Variable(y.float().cuda())
        recon_batch, mu, logvar = model(data, y)
        test_loss += loss_function(recon_batch, data, mu, logvar).data.cpu()[0]
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                  recon_batch.view(args.batch_size, 1, 32, 32)[:n]])
            save_image(comparison.data.cpu(),
                     args.result + '/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    if not os.path.exists(args.result):
        os.mkdir(args.result)
    train(epoch)
    test(epoch)
    #with torch.no_grad():
    sample = Variable(torch.randn(64, 100).cuda())
    trans = TargetTrans()
    for i in range(10):
        y = trans(i)
        y = torch.FloatTensor(y)
        y = y.repeat(64, 1).cuda()
        y = Variable(y)
        img = model.decode(sample, y).cpu()
        save_image(img.view(64, 1, 32, 32).data.cpu(),
                   args.result + '/sample_{}_'.format(i) + str(epoch) + '.png')
