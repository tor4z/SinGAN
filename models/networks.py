import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd)),
        self.add_module('norm', nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class WDiscriminator(nn.Module):
    def __init__(self, opt, nfc):
        super(WDiscriminator, self).__init__()
        self.is_cuda = opt.cuda
        N = nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt, nfc):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = opt.cuda
        N = nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1) #GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride =1, padding=opt.padd_size),
            nn.Tanh()
        )
        
    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        # Crop center of y fit to x
        y = y[:, :, ind:(y.shape[2]-ind), ind:(y.shape[3]-ind)]
        return x + y


def init_models(opt, nfc):
    #generator initialization:``
    netG = GeneratorConcatSkip2CleanAdd(opt, nfc)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    
    netG = netG.to(opt.device)
    if opt.devices:
        netG = nn.DataParallel(netG, device_ids=opt.devices)

    #discriminator initialization:
    netD = WDiscriminator(opt, nfc)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    
    netD = netD.to(opt.device)
    if opt.devices:
        netD = nn.DataParallel(netD, device_ids=opt.devices)

    return netD, netG

