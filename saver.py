import os
import shutil
import torch
import matplotlib.pyplot as plt

from SinGAN.functions import convert_image_np


class Saver(object):
    PYRAMID_NAME = 'pyramid.pth'
    NETG_NAME = 'netG.pth'
    NETD_NAME = 'netD.pth'

    def __init__(self, opt):
        self.opt = opt
        self.path = self.generate_dir2save()

        if os.path.exists(self.path):
            if not opt.force:
                print('trained model already exist')
                exit(0)
            else:
                shutil.rmtree(self.path)

        try:
            os.makedirs(self.path)
        except OSError:
            print('Can not make dir {}'.format(self.path))
            exit(1)

    def join(self, path, name):
        return os.path.join(path, name)

    def save(self, obj, path):
        torch.save(obj, path)

    def load(self, path):
        return torch.load(path)

    def generate_dir2save(self):
        dir2save = None
        opt = self.opt
        if (opt.mode == 'train') | (opt.mode == 'SR_train'):
            dir2save = 'TrainedModels/%s/scale_factor=%f,alpha=%d' % (opt.input_name[:-4], opt.scale_factor_init, opt.alpha)
        elif (opt.mode == 'animation_train') :
            dir2save = 'TrainedModels/%s/scale_factor=%f_noise_padding' % (opt.input_name[:-4], opt.scale_factor_init)
        elif (opt.mode == 'paint_train') :
            dir2save = 'TrainedModels/%s/scale_factor=%f_paint/start_scale=%d' % (opt.input_name[:-4], opt.scale_factor_init, opt.paint_start_scale)
        elif opt.mode == 'random_samples':
            dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out, opt.input_name[:-4], opt.gen_start_scale)
        elif opt.mode == 'random_samples_arbitrary_sizes':
            dir2save = '%s/RandomSamples_ArbitrerySizes/%s/scale_v=%f_scale_h=%f' % (opt.out, opt.input_name[:-4], opt.scale_v, opt.scale_h)
        elif opt.mode == 'animation':
            dir2save = '%s/Animation/%s' % (opt.out, opt.input_name[:-4])
        elif opt.mode == 'SR':
            dir2save = '%s/SR/%s' % (opt.out, opt.sr_factor)
        elif opt.mode == 'harmonization':
            dir2save = '%s/Harmonization/%s/%s_out' % (opt.out, opt.input_name[:-4], opt.ref_name[:-4])
        elif opt.mode == 'editing':
            dir2save = '%s/Editing/%s/%s_out' % (opt.out, opt.input_name[:-4], opt.ref_name[:-4])
        elif opt.mode == 'paint2image':
            dir2save = '%s/Paint2image/%s/%s_out' % (opt.out, opt.input_name[:-4], opt.ref_name[:-4])
            if opt.quantization_flag:
                dir2save = '%s_quantized' % dir2save
        return dir2save

    def load_trained_pyramid(self):
        path = self.join(self.path, self.PYRAMID_NAME)
        if(os.path.exists(path)):
            stat = self.load(path)
        else:
            print('no appropriate trained model is exist, please train first')
        return stat['Gs'], stat['Zs'], stat['reals'], stat['NoiseAmp']

    def save_pyramid(self, Gs, Zs, reals, NoiseAmp):
        stat = {
            'Gs': Gs,
            'Zs': Zs,
            'reals': reals,
            'NoiseAmp': NoiseAmp}
        path = self.join(self.path, self.PYRAMID_NAME)
        self.save(stat, path)

    def save_networks(self, netG, netD, z, scale_num):
        path = self.join(self.path, str(scale_num))

        if not os.path.exists(path):
            try:
                os.mkdir(path)
            except OSError:
                print('Can not create dir: {}'.format(path))

        self.save_network(netG, self.join(path, self.NETG_NAME))
        self.save_network(netD, self.join(path, self.NETD_NAME))
        # self.save(z, self.join(path, 'z_opt.pth'))

    def load_networks(self, netG, netD, scale_num):
        path = self.join(self.path, str(scale_num))
        if not os.path.exists(path):
            raise OSError('Can not find dir: {}'.format(path))

        netG = self.load_network(netG, self.join(path, self.NETG_NAME))
        netD = self.load_network(netD, self.join(path, self.NETD_NAME))
        return netG, netD

    def save_network(self, net, path):
        self.save(net.state_dict(), path)

    def load_network(self, net, path):
        state_dict = self.load(path)
        net.load_state_dict(state_dict)
        return net

    def imsave(self, tensor, path):
        img = functions.convert_image_np(tensor)
        plt.imsave(path, img, vmin=0, vmax=1)
