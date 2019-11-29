import os
import math
import torch
import torch.nn as nn

from .utils import functions
from .utils import imresize
from .layer_gan import LayerGAN


class SinGAN(nn.Module):
    def __init__(self, opt, summary, saver):
        super(SinGAN, self).__init__()
        self.opt = opt
        self.summary = summary
        self.saver = saver
        self.initialize()

    def initialize(self):
        self.Gs = []
        self.Zs = []
        self.NoiseAmp = []
        self.in_s = 0
        self.scale_num = 0
        self.reals = []
        self.nfc = self.opt.nfc_init
        self.min_nfc = self.opt.min_nfc_init
        self.create_reals_pyramid()
        self.total_scale = self.opt.stop_scale + 1

        self.summary.add_image('origin', self.raw_real, 0)

    def create_reals_pyramid(self):
        self.raw_real = functions.read_image(self.opt)
        if self.opt.mode == 'SR':
            functions.adjust_scales2image_SR(self.raw_real, self.opt)
        else:
            functions.adjust_scales2image(self.raw_real, self.opt)
        self.real = imresize.imresize(self.raw_real, self.opt.scale1, self.opt)
        return functions.creat_reals_pyramid(self.real, self.reals, self.opt)

    def train_pyramid(self):
        nfc_prev = 0

        while self.scale_num < self.total_scale:
            self.nfc = min(self.opt.nfc_init * pow(2, math.floor(self.scale_num / 4)), 128)
            self.min_nfc = min(self.opt.min_nfc_init * pow(2, math.floor(self.scale_num / 4)), 128)

            if self.opt.fast_training:
                if (self.scale_num > 0) & (self.scale_num % 4==0):
                    self.opt.niter = self.opt.niter // 2

            LG_curr = LayerGAN(self.opt, self)
            if (nfc_prev == self.nfc):
                LG_curr.load_network(self.scale_num-1)

            LG_curr.train_scale()
            LG_curr.save_networks()

            self.scale_num += 1
            nfc_prev = self.nfc

            self.visualize()
            self.save_pyramid()
            del LG_curr

    def save_pyramid(self):
        self.saver.save_pyramid(self.Gs, self.Zs, self.reals, self.NoiseAmp)

    def load_pyramid(self):
        self.Gs, self.Zs, self.reals, self.NoiseAmp = self.saver.load_pyramid()

    def visualize(self):
        self.summary.add_image('scale/{}/real'.format(self.scale_num), self.reals[self.scale_num - 1], 0)

    def generate(self, in_s=None, scale_v=1, scale_h=1, n=0, gen_start_scale=0, num_samples=50):
        #if torch.is_tensor(in_s) == False:
        if in_s is None:
            in_s = torch.full(self.reals[0].shape, 0, device=self.opt.device)
        images_cur = []
        for G, Z_opt, noise_amp in zip(self.Gs, self.Zs, self.NoiseAmp):
            pad1 = ((self.opt.ker_size - 1) * self.opt.num_layer) / 2
            m = nn.ZeroPad2d(int(pad1))
            nzx = (Z_opt.shape[2] - pad1 * 2) * scale_v
            nzy = (Z_opt.shape[3] - pad1 * 2) * scale_h

            images_prev = images_cur
            images_cur = []

            for i in range(0, num_samples, 1):
                if n == 0:
                    z_curr = functions.generate_noise([1, nzx, nzy], device=self.opt.device)
                    z_curr = z_curr.expand(1, 3, z_curr.shape[2], z_curr.shape[3])
                    z_curr = m(z_curr)
                else:
                    z_curr = functions.generate_noise([self.opt.nc_z, nzx, nzy], device=self.opt.device)
                    z_curr = m(z_curr)

                if images_prev == []:
                    I_prev = m(in_s)
                else:
                    I_prev = images_prev[i]
                    I_prev = imresize.imresize(I_prev, 1 / self.opt.scale_factor, self.opt)
                    if self.opt.mode != "SR":
                        I_prev = I_prev[:, :, 0:round(scale_v * self.reals[n].shape[2]), 0:round(scale_h * self.reals[n].shape[3])]
                        I_prev = m(I_prev)
                        I_prev = I_prev[:, :, 0:z_curr.shape[2], 0:z_curr.shape[3]]
                        I_prev = functions.upsampling(I_prev, z_curr.shape[2], z_curr.shape[3])
                    else:
                        I_prev = m(I_prev)

                if n < gen_start_scale:
                    z_curr = Z_opt

                z_in = noise_amp * (z_curr) + I_prev
                I_curr = G(z_in.detach(), I_prev)

                if n == len(self.reals) - 1:
                    if self.opt.mode == 'train':
                        dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (self.opt.out, self.opt.input_name[:-4], gen_start_scale)
                    else:
                        dir2save = functions.generate_dir2save(self.opt)
                    try:
                        os.makedirs(dir2save)
                    except OSError:
                        pass
                    if (self.opt.mode != "harmonization") & (self.opt.mode != "editing") & (self.opt.mode != "SR") & (self.opt.mode != "paint2image"):
                        self.summary.add_image('{}/{}'.format(dir2save, i), I_curr.detach(), 0)
                        #plt.imsave('%s/%d_%d.png' % (dir2save,i,n),functions.convert_image_np(I_curr.detach()), vmin=0, vmax=1)
                        #plt.imsave('%s/in_s.png' % (dir2save), functions.convert_image_np(in_s), vmin=0,vmax=1)
                images_cur.append(I_curr)
            n += 1
        print('generate end.')
        return I_curr.detach()

    def generate_gif(self, Gs,Zs,reals,NoiseAmp,opt,alpha=0.1,beta=0.9,start_scale=2,fps=10):

        in_s = torch.full(Zs[0].shape, 0, device=self.opt.device)
        images_cur = []
        count = 0

        for G, Z_opt, noise_amp, real in zip(Gs, Zs, NoiseAmp, reals):
            pad_image = int(((self.opt.ker_size - 1) * self.opt.num_layer) / 2)
            nzx = Z_opt.shape[2]
            nzy = Z_opt.shape[3]
            #pad_noise = 0
            #m_noise = nn.ZeroPad2d(int(pad_noise))
            m_image = nn.ZeroPad2d(int(pad_image))
            images_prev = images_cur
            images_cur = []
            if count == 0:
                z_rand = functions.generate_noise([1, nzx, nzy], device=self.opt.device)
                z_rand = z_rand.expand(1, 3, Z_opt.shape[2], Z_opt.shape[3])
                z_prev1 = 0.95 * Z_opt +0.05 * z_rand
                z_prev2 = Z_opt
            else:
                z_prev1 = 0.95 * Z_opt + 0.05 * functions.generate_noise([self.opt.nc_z, nzx, nzy], device=self.opt.device)
                z_prev2 = Z_opt

            for i in range(0, 100, 1):
                if count == 0:
                    z_rand = functions.generate_noise([1, nzx, nzy], device=self.opt.device)
                    z_rand = z_rand.expand(1, 3, Z_opt.shape[2], Z_opt.shape[3])
                    diff_curr = beta * (z_prev1 - z_prev2) + (1 - beta) * z_rand
                else:
                    diff_curr = beta * (z_prev1 - z_prev2) + (1 - beta) * (functions.generate_noise([self.opt.nc_z, nzx, nzy], device=self.opt.device))

                z_curr = alpha * Z_opt + (1 - alpha) * (z_prev1 + diff_curr)
                z_prev2 = z_prev1
                z_prev1 = z_curr

                if images_prev == []:
                    I_prev = in_s
                else:
                    I_prev = images_prev[i]
                    I_prev = imresize(I_prev, 1 / self.opt.scale_factor, self.opt)
                    I_prev = I_prev[:, :, 0:real.shape[2], 0:real.shape[3]]
                    #I_prev = functions.upsampling(I_prev,reals[count].shape[2],reals[count].shape[3])
                    I_prev = m_image(I_prev)
                if count < start_scale:
                    z_curr = Z_opt

                z_in = noise_amp * z_curr + I_prev
                I_curr = G(z_in.detach(), I_prev)

                if (count == len(Gs)-1):
                    I_curr = functions.denorm(I_curr).detach()
                    I_curr = I_curr[0, :, :, :].cpu().numpy()
                    I_curr = I_curr.transpose(1, 2, 0) * 255
                    I_curr = I_curr.astype(np.uint8)

                images_cur.append(I_curr)
            count += 1
        dir2save = functions.generate_dir2save(self.opt)
        try:
            os.makedirs('%s/start_scale=%d' % (dir2save, start_scale) )
        except OSError:
            pass
        imageio.mimsave('%s/start_scale=%d/alpha=%f_beta=%f.gif' % (dir2save, start_scale, alpha, beta), images_cur, fps=fps)
        del images_cur

    def train_paint(self, opt, Gs, Zs, reals, NoiseAmp, centers, paint_inject_scale):
        in_s = torch.full(reals[0].shape, 0, device=self.opt.device)
        scale_num = 0
        nfc_prev = 0

        while scale_num < self.opt.stop_scale + 1:
            if scale_num != paint_inject_scale:
                scale_num += 1
                nfc_prev = self.nfc
                continue
            else:
                self.opt.nfc = min(self.opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
                self.opt.min_nfc = min(self.opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

                self.opt.out_ = functions.generate_dir2save(opt)
                self.opt.outf = '%s/%d' % (self.opt.out_, scale_num)
                try:
                    os.makedirs(self.opt.outf)
                except OSError:
                        pass

                #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
                #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
                self.summary.add_image('{}/in_scale'.format(self.opt.input_name), reals[scale_num], 0)

                D_curr,G_curr = init_models(self.opt)

                z_curr, in_s, G_curr = train_single_scale(D_curr, G_curr, reals[:scale_num+1], Gs[:scale_num], Zs[:scale_num], in_s, NoiseAmp[:scale_num], opt, centers=centers)

                G_curr = functions.reset_grads(G_curr, False)
                G_curr.eval()
                D_curr = functions.reset_grads(D_curr, False)
                D_curr.eval()

                Gs[scale_num] = G_curr
                Zs[scale_num] = z_curr
                NoiseAmp[scale_num] = self.opt.noise_amp

                torch.save(Zs, '%s/Zs.pth' % (self.opt.out_))
                torch.save(Gs, '%s/Gs.pth' % (self.opt.out_))
                torch.save(reals, '%s/reals.pth' % (self.opt.out_))
                torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (self.opt.out_))

                scale_num += 1
                nfc_prev = self.nfc
            del D_curr, G_curr


