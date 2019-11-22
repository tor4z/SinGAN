import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

import .networks as networks
import .functions as functions


class ScaleGAN(nn.Module):
    def __init__(self, opt, sin_gan):
        super(ScaleGAN, self).__init__()

        pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
        pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)

        self.m_noise = nn.ZeroPad2d(int(pad_noise))
        self.m_image = nn.ZeroPad2d(int(pad_image))

        self.opt = opt
        self.netD, self.netG = networks.init_models(self.opt, self.sin_gan.nfc)

        # setup optimizer
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.opt.lr_d, betas=(self.opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.opt.lr_g, betas=(self.opt.beta1, 0.999))
        self.schedulerD = MultiStepLR(optimizer=self.optimizerD, milestones=[1600], gamma=self.opt.gamma)
        self.schedulerG = MultiStepLR(optimizer=self.optimizerG, milestones=[1600], gamma=self.opt.gamma)

        self.criterionMSE = nn.MSELoss()

        self.sin_gan = sin_gan
        
    def initialize(self):
        self.real = self.sin_gan.reals[len(self.sin_gan.Gs)]
        self.opt.nzx = self.real.shape[2]
        self.opt.nzy = self.real.shape[3]
        self.noise_amp = 1

        if self.opt.mode == 'animation_train':
            self.opt.nzx = self.real.shape[2] + (self.opt.ker_size - 1) * (self.opt.num_layer)
            self.opt.nzy = self.real.shape[3] + (self.opt.ker_size - 1) * (self.opt.num_layer)
            self.pad_noise = 0

    def train_G(self):
        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################
        for j in range(self.opt.Gsteps):
            self.netG.zero_grad()
            output = self.netD(fake)
            #D_fake_map = output.detach()
            errG = -output.mean()
            errG.backward(retain_graph=True)
            if alpha!=0:
                loss = nn.MSELoss()
                if opt.mode == 'paint_train':
                    z_prev = functions.quant2centers(z_prev, centers, opt)
                    plt.imsave('%s/z_prev.png' % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)
                Z_opt = opt.noise_amp * z_opt + z_prev
                rec_loss = alpha * loss(self.netG(Z_opt.detach(), z_prev), self.real)
                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()
            else:
                Z_opt = z_opt
                rec_loss = 0

            self.optimizerG.step()

    def train_D(self):
        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(self.opt.Dsteps):
            # train with real
            self.netD.zero_grad()

            output = self.netD(real)
            errD_real = -output.mean()
            errD_real.backward(retain_graph=True)
            D_x = -errD_real.item()

            # train with fake
            if (j==0) & (epoch == 0):
                if (self.sin_gan.Gs == []) & (self.opt.mode != 'SR_train'):
                    prev = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    in_s = prev
                    prev = m_image(prev)
                    z_prev = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    z_prev = m_noise(z_prev)
                    self.noise_amp = 1
                elif opt.mode == 'SR_train':
                    z_prev = in_s
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    self.noise_amp = opt.noise_amp_init * RMSE
                    z_prev = m_image(z_prev)
                    prev = z_prev
                else:
                    prev = self.draw('rand')
                    prev = self.m_image(prev)
                    z_prev = self.draw('rec')
                    RMSE = torch.sqrt(self.criterionMSE(real, z_prev))
                    self.noise_amp = opt.noise_amp_init * RMSE
                    z_prev = self.m_image(z_prev)
            else:
                prev = self.draw('rand')
                prev = self.m_image(prev)

            if opt.mode == 'paint_train':
                prev = functions.quant2centers(prev, centers, opt)
                plt.imsave('%s/prev.png' % (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)

            if (Gs == []) & (opt.mode != 'SR_train'):
                noise = noise_
            else:
                noise = opt.noise_amp * noise_ + prev

            fake = self.netG(noise.detach(), prev)
            output = self.netD(fake.detach())
            errD_fake = output.mean()
            errD_fake.backward(retain_graph=True)
            D_G_z = output.mean().item()

            gradient_penalty = self.gradient_penalty(real, fake)
            gradient_penalty.backward()

            errD = errD_real + errD_fake + gradient_penalty
            optimizerD.step()

    def scheduler_step(self):
        self.schedulerD.step()
        self.schedulerG.step()

    def train_scale(self):
        fixed_noise = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
        z_opt = torch.full(fixed_noise.shape, 0, device=opt.device)
        z_opt = m_noise(z_opt)

        for epoch in range(opt.niter):
            self.epoch_curr = epoch
            if (Gs == []) & (opt.mode != 'SR_train'):
                z_opt = functions.generate_noise([1, opt.nzx, opt.nzy], device=opt.device)
                z_opt = self.m_noise(z_opt.expand(1, 3, opt.nzx, opt.nzy))
                noise_ = functions.generate_noise([1, opt.nzx, opt.nzy], device=opt.device)
                noise_ = self.m_noise(noise_.expand(1, 3, opt.nzx, opt.nzy))
            else:
                noise_ = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
                noise_ = self.m_noise(noise_)

            self.train_D()
            self.train_G()
            self.scheduler_step()

    def save_networks(self):
        self.sin_gan.saver.save_networks(self.netG, self.netD, self.scale_num)

    def load_network(self, scale_num):
        self.sin_gan.save.load_networks(self.netG, self.netD, scale_num)

    def reset_grads(self, model, require_grad):
        for p in model.parameters():
            p.requires_grad_(require_grad)
        return model

    def eval(self):
        self.G_curr = self.reset_grads(self.G_curr, False)
        self.G_curr.eval()
        self.D_curr = self.reset_grads(self.D_curr, False)
        self.D_curr.eval()

    @property
    def global_steps(self):
        return self.scale_num * self.opt.niter + self.epoch_curr

    def draw(self, mode):
        G_z = in_s
        if len(Gs) > 0:
            if mode == 'rand':
                count = 0
                pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
                if opt.mode == 'animation_train':
                    pad_noise = 0
                for G, Z_opt, real_curr, real_next, noise_amp in zip(self.Gs, self.Zs, self.reals, self.reals[1:], self.NoiseAmp):
                    if count == 0:
                        z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                        z = z.expand(1, 3, z.shape[2], z.shape[3])
                    else:
                        z = functions.generate_noise([opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                    z = self.m_noise(z)
                    G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                    G_z = self.m_image(G_z)
                    z_in = noise_amp * z + G_z
                    G_z = G(z_in.detach(), G_z)
                    G_z = imresize(G_z, 1/opt.scale_factor, opt)
                    G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                    count += 1
            if mode == 'rec':
                count = 0
                for G, Z_opt, real_curr, real_next, noise_amp in zip(self.Gs, self.Zs, self.reals, self.reals[1:], self.NoiseAmp):
                    G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                    G_z = self.m_image(G_z)
                    z_in = noise_amp * Z_opt + G_z
                    G_z = G(z_in.detach(), G_z)
                    G_z = imresize(G_z,1/opt.scale_factor, opt)
                    G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                    count += 1
        return G_z

    def gradient_penalty(real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(self.opt.device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.netD(interpolates)
        grad_outputs = torch.ones(disc_interpolates.size()).to(self.opt.device)

        gradients = torch.autograd.grad(outputs=disc_interpolates,
                                        inputs=interpolates,
                                        grad_outputs=grad_outputs,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        #LAMBDA = 1
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.opt.lambda_grad

    def visualize(self):
        if self.global_steps % opt.v_freq == 0:
            summary = self.sin_gan.summary
            summary.add_image('scale/{}/real'.format(scale_num), reals[scale_num], 0)
            summary.add_image('fake_sample/train_scale_{}'.format(scale_num), fake, self.global_steps)
            summary.add_image('G(z_opt)/train_scale_{}'.format(scale_num),  netG(Z_opt.detach(), z_prev), self.global_steps)
            summary.add_image('z_opt/train_scale_{}'.format(scale_num), z_opt, self.global_steps)
            summary.add_image('prev/train_scale_{}'.format(scale_num), prev, self.global_steps)
            summary.add_image('noise/train_scale_{}'.format(scale_num), noise, self.global_steps)
            summary.add_image('z_prev/train_scale_{}'.format(scale_num), z_prev, self.global_steps)

            summary.add_scalars('main', {'errD': errD.detach().item(),
                                'errG': (errG.detach() + rec_loss).item(),
                                'D_real': D_x,
                                'D_fake': D_G_z,
                                'z_opt': rec_loss.detach().item(),}, self.global_steps)
