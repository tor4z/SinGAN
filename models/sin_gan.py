import os
import math
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt

import .functions as functions
import .network as network
from .imresize import imresize
from .scale_gan import ScaleGAN


class SinGAN(nn.Module):
    def __init__(self, opt, summary, saver):
        super(SinGAN, self).__init__()
        self.opt = opt
        self.summary = summary
        self.saver = saver
        initialize()

    def initialize(self):
        self.Gs = []
        self.Zs = []
        self.NoiseAmp = []
        self.in_s = 0
        self.scale_num = 0
        self.nfc = self.opt.nfc_init
        self.min_nfc = self.opt.min_nfc_init
        self.reals = self.create_reals_pyramid()

        self.summary.add_image('origin', self.raw_real, 0)

    def create_reals_pyramid(self):
        self.raw_real = functions.read_image(opt)
        self.real = imresize(self.raw_real, opt.scale1, opt)
        return functions.creat_reals_pyramid(real, reals, opt)

    def train_pyramid(self):
        nfc_prev = 0

        while self.scale_num < self.opt.stop_scale + 1:
            self.nfc = min(self.opt.nfc_init * pow(2, math.floor(self.scale_num / 4)), 128)
            self.min_nfc = min(self.opt.min_nfc_init * pow(2, math.floor(self.scale_num / 4)), 128)

            if self.opt.fast_training:
                if (self.scale_num > 0) & (self.scale_num % 4==0):
                    self.opt.niter = self.opt.niter // 2

            SG_curr = ScaleGAN(opt, self)
            if (nfc_prev == opt.nfc):
                SG_curr.load_network(self.scale_num-1)

            SG_curr.train_scale()
            SG_curr.save_networks()

            self.scale_num += 1
            nfc_prev = self.nfc

            self.visualize()
            self.save_pyramid()

    def save_pyramid(self):
        self.saver.save_pyramid(self.Gs, self.Zs, self.reals, self.NoiseAmp)

    def load_pyramid(self):
        self.Gs, self.Zs, self.reals, self.NoiseAmp = self.saver.load_pyramid()

    def visualize(self):
        self.summary.add_image('scale/{}/real'.format(self.scale_num), self.reals[self.scale_num], 0)


def train_paint(opt, Gs, Zs, reals, NoiseAmp, centers, paint_inject_scale):
    in_s = torch.full(reals[0].shape, 0, device=opt.device)
    scale_num = 0
    nfc_prev = 0

    while scale_num < opt.stop_scale + 1:
        if scale_num != paint_inject_scale:
            scale_num += 1
            nfc_prev = opt.nfc
            continue
        else:
            opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

            opt.out_ = functions.generate_dir2save(opt)
            opt.outf = '%s/%d' % (opt.out_, scale_num)
            try:
                os.makedirs(opt.outf)
            except OSError:
                    pass

            #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
            #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
            plt.imsave('%s/in_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

            D_curr,G_curr = init_models(opt)

            z_curr, in_s, G_curr = train_single_scale(D_curr, G_curr, reals[:scale_num+1], Gs[:scale_num], Zs[:scale_num], in_s, NoiseAmp[:scale_num], opt, centers=centers)

            G_curr = functions.reset_grads(G_curr, False)
            G_curr.eval()
            D_curr = functions.reset_grads(D_curr, False)
            D_curr.eval()

            Gs[scale_num] = G_curr
            Zs[scale_num] = z_curr
            NoiseAmp[scale_num] = opt.noise_amp

            torch.save(Zs, '%s/Zs.pth' % (opt.out_))
            torch.save(Gs, '%s/Gs.pth' % (opt.out_))
            torch.save(reals, '%s/reals.pth' % (opt.out_))
            torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

            scale_num += 1
            nfc_prev = opt.nfc
        del D_curr, G_curr


