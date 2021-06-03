from model.BasicModule import BasicModule
import torch.nn as nn


class Generator(BasicModule):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.model_name = "NetG"
        ngf = opt.ngf
        self.main = nn.Sequential(
            nn.ConvTranspose2d(opt.nz, ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf, 3, 5, 3, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(BasicModule):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.model_name = 'NetD'
        ndf = opt.ndf
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 5, 3, 1),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0)
        )

    def forward(self, x):
        return self.main(x).view(-1)