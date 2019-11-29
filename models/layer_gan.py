import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import trange

from . import networks
from .utils import functions, imresize


class LayerGAN(nn.Module):
    def __init__(self, opt, sin_gan):
        super(LayerGAN, self).__init__()
        self.sin_gan = sin_gan
        self.opt = opt

        pad_noise = int(((self.opt.ker_size - 1) * self.opt.num_layer) / 2)
        pad_image = int(((self.opt.ker_size - 1) * self.opt.num_layer) / 2)

        self.m_noise = nn.ZeroPad2d(int(pad_noise))
        self.m_image = nn.ZeroPad2d(int(pad_image))

        self.netD, self.netG = networks.init_models(self.opt, self.sin_gan.nfc)
        # setup optimizer
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.opt.lr_d, betas=(self.opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.opt.lr_g, betas=(self.opt.beta1, 0.999))
        self.schedulerD = MultiStepLR(optimizer=self.optimizerD, milestones=[1600], gamma=self.opt.gamma)
        self.schedulerG = MultiStepLR(optimizer=self.optimizerG, milestones=[1600], gamma=self.opt.gamma)

        self.criterionMSE = nn.MSELoss()
        self.initialize()

    def initialize(self):
        self.real = self.sin_gan.reals[len(self.sin_gan.Gs)]
        self.nzx = self.real.shape[2]
        self.nzy = self.real.shape[3]
        self.nc_z = self.opt.nc_z
        self.mode = self.opt.mode
        self.device = self.opt.device
        self.alpha = self.opt.alpha
        self.noise_amp = 1
        self.Dsteps = self.opt.Dsteps
        self.Gsteps = self.opt.Gsteps
        self.scale_num = self.sin_gan.scale_num

        if self.mode == 'animation_train':
            self.nzx = self.real.shape[2] + (self.opt.ker_size - 1) * (self.opt.num_layer)
            self.nzy = self.real.shape[3] + (self.opt.ker_size - 1) * (self.opt.num_layer)
            self.pad_noise = 0

        fixed_noise = functions.generate_noise([self.nc_z, self.nzx, self.nzy],
                                                device=self.device)
        z_opt = torch.full(fixed_noise.shape, 0, device=self.device)
        self.z_opt = self.m_noise(z_opt)

    def scheduler_step(self):
        self.schedulerD.step()
        self.schedulerG.step()

    def train_D(self, noise_):
        for j in range(self.Dsteps):
            self.netD.zero_grad()

            output = self.netD(self.real)
            errD_real = -output.mean()
            errD_real.backward(retain_graph=True)
            self.lossD_real = -errD_real.detach()

            if (j==0) & (self.epoch_curr == 0):
                if (self.sin_gan.Gs == []) & (self.mode != 'SR_train'):
                    prev = torch.full([1, self.nc_z, self.nzx, self.nzy], 0, device=self.device)
                    self.sin_gan.in_s = prev
                    prev = self.m_image(prev)
                    self.z_prev = torch.full([1, self.nc_z, self.nzx, self.nzy], 0, device=self.device)
                    self.z_prev = self.m_noise(self.z_prev)
                    self.noise_amp = 1
                elif self.opt.mode == 'SR_train':
                    self.z_prev = self.sin_gan.in_s
                    RMSE = torch.sqrt(self.criterionMSE(self.real, self.z_prev))
                    self.noise_amp = self.opt.noise_amp_init * RMSE
                    self.z_prev = self.m_image(self.z_prev)
                    prev = self.z_prev
                else:
                    prev = self.draw('rand')
                    prev = self.m_image(prev)
                    self.z_prev = self.draw('rec')
                    RMSE = torch.sqrt(self.criterionMSE(self.real, self.z_prev))
                    self.noise_amp = self.opt.noise_amp_init * RMSE
                    self.z_prev = self.m_image(self.z_prev)
            else:
                prev = self.draw('rand')
                prev = self.m_image(prev)

            if self.mode == 'paint_train':
                prev = functions.quant2centers(prev, centers, self.opt)
                plt.imsave('%s/prev.png' % (self.opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)

            if (self.sin_gan.Gs == []) & (self.mode != 'SR_train'):
                noise = noise_
            else:
                # Inject noise in every step
                noise = self.noise_amp * noise_ + prev

            fake = self.netG(noise.detach(), prev)
            output = self.netD(fake.detach())
            errD_fake = output.mean()
            errD_fake.backward(retain_graph=True)
            self.lossD_fake = errD_fake.detach()
            

            gradient_penalty = self.criterionGRAD(self.real, fake)
            gradient_penalty.backward()
            self.lossGrad = gradient_penalty.detach()

            self.lossD = (errD_real + errD_fake + gradient_penalty).detach()
            self.optimizerD.step()
        return fake

    def train_G(self, fake):
        for j in range(self.Gsteps):
            # discriminator fake
            self.netG.zero_grad()
            output = self.netD(fake)
            errG = -output.mean()
            errG.backward(retain_graph=True)
            self.lossG = errG.detach()

            # alpha reconstruction loss weight
            # if alpha is 0, mean no reconstruction
            if self.alpha != 0:
                if self.mode == 'paint_train':
                    self.z_prev = functions.quant2centers(self.z_prev, centers, self.opt)
                    plt.imsave('%s/z_prev.png' % (self.opt.outf), functions.convert_image_np(self.z_prev), vmin=0, vmax=1)
                z_opt = self.noise_amp * self.z_opt + self.z_prev
                # the distance between fake and real
                # minimize the loss to improve the reconstruction ability
                rec_loss = self.alpha * self.criterionMSE(self.netG(z_opt.detach(), self.z_prev), self.real)
                rec_loss.backward(retain_graph=True)
                self.lossRec = rec_loss.detach()
            else:
                self.lossRec = 0
            self.optimizerG.step()

    def train_scale(self):
        with trange(self.opt.niter) as t:
            t.set_description('scale: {}/{} '.format(self.scale_num, self.sin_gan.total_scale))
            for epoch in t:
                self.epoch_curr = epoch
                if (self.sin_gan.Gs == []) & (self.mode != 'SR_train'):
                    self.z_opt = functions.generate_noise([1, self.nzx, self.nzy], device=self.device)
                    self.z_opt = self.m_noise(self.z_opt.expand(1, 3, self.nzx, self.nzy))
                    noise_ = functions.generate_noise([1, self.nzx, self.nzy], device=self.device)
                    noise_ = self.m_noise(noise_.expand(1, 3, self.nzx, self.nzy))
                else:
                    noise_ = functions.generate_noise([self.nc_z, self.nzx, self.nzy], device=self.device)
                    noise_ = self.m_noise(noise_)

                fake = self.train_D(noise_)
                self.train_G(fake)

                self.scheduler_step()
                self.visualize(fake)
            self.save_networks()
            self.save_layer()

    def save_layer(self):
        self.eval()
        self.sin_gan.Gs.append(self.netG)
        self.sin_gan.Zs.append(self.z_opt)
        self.sin_gan.NoiseAmp.append(self.noise_amp)

    def save_networks(self):
        self.sin_gan.saver.save_networks(self.netG, self.netD, self.scale_num)

    def load_network(self, scale_num):
        self.sin_gan.saver.load_networks(self.netG, self.netD, scale_num)

    def reset_grads(self, model, require_grad):
        for p in model.parameters():
            p.requires_grad_(require_grad)
        return model

    def eval(self):
        self.netG = self.reset_grads(self.netG, False)
        self.netG.eval()
        self.netD = self.reset_grads(self.netD, False)
        self.netD.eval()

    def global_steps(self):
        return self.scale_num * self.opt.niter + self.epoch_curr

    def draw(self, mode):
        G_z = self.sin_gan.in_s
        if len(self.sin_gan.Gs) > 0:
            if mode == 'rand':
                count = 0
                pad_noise = int(((self.opt.ker_size - 1) * self.opt.num_layer) / 2)
                if self.mode == 'animation_train':
                    pad_noise = 0
                for G, Z_opt, real_curr, real_next, noise_amp in zip(self.sin_gan.Gs,
                                                                     self.sin_gan.Zs,
                                                                     self.sin_gan.reals,
                                                                     self.sin_gan.reals[1:],
                                                                     self.sin_gan.NoiseAmp):
                    if count == 0:
                        z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise],
                                                     device=self.opt.device)
                        z = z.expand(1, 3, z.shape[2], z.shape[3])
                    else:
                        z = functions.generate_noise([self.nc_z, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise],
                                                     device=self.opt.device)
                    z = self.m_noise(z)                                                            # Padding noise
                    G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]                    # Crop
                    G_z = self.m_image(G_z)                                                        # Padding img
                    z_in = noise_amp * z + G_z                                                     # Make z
                    G_z = G(z_in.detach(), G_z)                                                    # Generate
                    G_z = imresize.imresize(G_z, 1/self.opt.scale_factor, self.opt)                # Resize to next
                    G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]                    # Crop to next
                    count += 1
            if mode == 'rec':
                count = 0
                for G, Z_opt, real_curr, real_next, noise_amp in zip(self.sin_gan.Gs,
                                                                     self.sin_gan.Zs,
                                                                     self.sin_gan.reals,
                                                                     self.sin_gan.reals[1:],
                                                                     self.sin_gan.NoiseAmp):
                    G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]                 # Crop
                    G_z = self.m_image(G_z)                                                     # Padding
                    z_in = self.noise_amp * Z_opt + G_z                                         # Make z
                    G_z = G(z_in.detach(), G_z)                                                 # Generate
                    G_z = imresize.imresize(G_z, 1/self.opt.scale_factor, self.opt)             # Resize to next
                    G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]                 # Crop to next   
                    count += 1
        return G_z

    def criterionGRAD(self, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(self.device)

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
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.opt.lambda_grad

    def visualize(self, fake):
        if self.global_steps() % self.opt.v_freq == 0:
            summary = self.sin_gan.summary
            scale_num = self.scale_num

            summary.add_image('scale/{}/real'.format(scale_num), self.sin_gan.reals[scale_num], 0)
            summary.add_image('fake_sample/train_scale_{}'.format(scale_num), fake, self.global_steps())
            # summary.add_image('G(z_opt)/train_scale_{}'.format(scale_num),  self.netG(Z_opt, z_prev), self.global_steps())
            summary.add_image('z_opt/train_scale_{}'.format(scale_num), self.z_opt, self.global_steps())
            # summary.add_image('prev/train_scale_{}'.format(scale_num), prev, self.global_steps())
            # summary.add_image('noise/train_scale_{}'.format(scale_num), noise, self.global_steps())
            summary.add_image('z_prev/train_scale_{}'.format(scale_num), self.z_prev, self.global_steps())

            summary.add_scalars('main', {'lossD': self.lossD.item(),
                                         'lossG': self.lossG.item(),
                                         'lossGrad': self.lossGrad.item(),
                                         'lossD_real': self.lossD_real.item(),
                                         'lossD_fake': self.lossD_fake.item(),
                                         'lossRec': self.lossRec.item()}, self.global_steps())
