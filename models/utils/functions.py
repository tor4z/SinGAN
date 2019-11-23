import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch.nn as nn
import scipy.io as sio
import math
from skimage import io as img
from skimage import color, morphology, filters
from .imresize import imresize
import os
import random
from sklearn.cluster import KMeans
from .utils import move_to_gpu, move_to_cpu, norm, denorm


def read_image(opt):
    x = img.imread('%s%s' % (opt.input_img, opt.ref_image))
    return np2torch(x)

def convert_image_np(inp):
    if inp.shape[1] == 3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1, :, :, :])
        inp = inp.numpy().transpose((1, 2, 0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1, -1, :, :])
        inp = inp.numpy().transpose((0, 1))

    inp = np.clip(inp, 0, 1)
    return inp

def convert_image_np_2d(inp):
    inp = denorm(inp)
    inp = inp.numpy()
    # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
    # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])
    # inp = std*
    return inp

def generate_noise(size, num_samp=1, device='cuda', type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = upsampling(noise, size[1], size[2])
    if type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device) + 5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1 + noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    if type == 'uniform+poisson':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise2 = np.random.poisson(10, [num_samp, int(size[0]), int(size[1]), int(size[2])])
        noise2 =  torch.from_numpy(noise2).to(device)
        if opt.cuda:
            noise2 = noise2.type(torch.cuda.FloatTensor)
        else:
            noise2 = noise2.type(torch.FloatTensor)
        noise = noise1 + noise2
    if type == 'poisson':
        noise = np.random.poisson(0.1, [num_samp, int(size[0]), int(size[1]), int(size[2])])
        noise =  torch.from_numpy(noise).to(device)
        if opt.cuda:
            noise = noise.type(torch.cuda.FloatTensor)
        else:
            noise = noise.type(torch.FloatTensor)
    return noise

def plot_learning_curves(G_loss, D_loss, epochs, label1, label2, name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n, G_loss, n, D_loss)
    #plt.title('loss')
    #plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend([label1,label2], loc='upper right')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def plot_learning_curve(loss, epochs, name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,loss)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def upsampling(im, sx, sy):
    m = nn.Upsample(size=[round(sx), round(sy)], mode='bilinear', align_corners=True)
    return m(im)

def read_image(opt):
    x = img.imread('%s/%s' % (opt.input_dir, opt.input_name))
    x = np2torch(x, opt)
    x = x[:, 0:3, :, :]
    return x

def read_image_dir(dir, opt):
    x = img.imread('%s' % (dir))
    x = np2torch(x, opt)
    x = x[:, 0:3, :, :]
    return x

def np2torch(x, opt):
    if opt.nc_im == 3:
        x = x[:, :, :, None]
        x = x.transpose((3, 2, 0, 1)) / 255
    else:
        x = color.rgb2gray(x)
        x = x[:, :, None, None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    x = move_to_gpu(x, opt)
    if opt.cuda:
        x = x.type(torch.cuda.FloatTensor)
    else:
        x = x.type(torch.FloatTensor)
    x = norm(x)
    return x

def torch2uint8(x):
    x = x[0, :, :, :]
    x = x.permute((1, 2, 0))
    x = 255 * denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x

def read_image2np(opt):
    x = img.imread('%s/%s' % (opt.input_dir, opt.input_name))
    x = x[:, :, 0:3]
    return x

def adjust_scales2image(real_, opt):
    opt.num_scales = int((math.log(math.pow(opt.min_size / (real_.shape[2]), 1), opt.scale_factor_init))) + 1
    scale2stop = int(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]), 1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = imresize(real_, opt.scale1, opt)
    opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    # opt.scale_factor = math.pow(opt.min_size/(min(real_.shape[0],real_.shape[1])),1/(opt.stop_scale))
    scale2stop = int(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real

def adjust_scales2image_SR(real_, opt):
    #'''
    opt.min_size = 18
    opt.num_scales = int((math.log(math.pow(opt.min_size / (min([real_.shape[2],real_.shape[3]])), 1), opt.scale_factor_init))) + 1
    scale2stop = int(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]), 1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = imresize(real_, opt.scale1, opt)
    opt.scale_factor = math.pow(opt.min_size / (min([real_.shape[2], real_.shape[3]])), 1 / (opt.stop_scale))
    # opt.scale_factor = math.pow(opt.min_size/(min(real_.shape[0],real_.shape[1])),1/(opt.stop_scale))
    scale2stop = int(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real

def creat_reals_pyramid(real, reals, opt):
    real = real[:, 0:3, :, :]
    for i in range(0, opt.stop_scale+1, 1):
        scale = math.pow(opt.scale_factor, opt.stop_scale - i)
        curr_real = imresize(real, scale, opt)
        reals.append(curr_real)
    return reals

def generate_in2coarsest(reals, scale_v, scale_h, opt):
    real = reals[opt.gen_start_scale]
    real_down = upsampling(real, scale_v * real.shape[2], scale_h * real.shape[3])
    if opt.gen_start_scale == 0:
        in_s = torch.full(real_down.shape, 0, device=opt.device)
    else: #if n!=0
        in_s = upsampling(real_down, real_down.shape[2], real_down.shape[3])
    return in_s

def calc_init_scale(opt):
    in_scale = math.pow(1/2,1/3)
    iter_num = round(math.log(1 / opt.sr_factor, in_scale))
    in_scale = pow(opt.sr_factor, 1 / iter_num)
    return in_scale, iter_num

def quant(prev, opt):
    arr = prev.reshape((-1, 3)).cpu()
    kmeans = KMeans(n_clusters=5, random_state=0).fit(arr)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    x = centers[labels]
    x = torch.from_numpy(x)
    x = move_to_gpu(x, opt)
    if opt.cuda:
        x = x.type(torch.cuda.FloatTensor)
    else:
        x = x.type(torch.FloatTensor)
    x = x.view(prev.shape)
    return x, centers

def quant2centers(paint, centers, opt):
    arr = paint.reshape((-1, 3)).cpu()
    kmeans = KMeans(n_clusters=5, init=centers, n_init=1).fit(arr)
    labels = kmeans.labels_
    #centers = kmeans.cluster_centers_
    x = centers[labels]
    x = torch.from_numpy(x)
    x = move_to_gpu(x, opt)
    if opt.cuda:
        x = x.type(torch.cuda.FloatTensor)
    else:
        x = x.type(torch.FloatTensor)
    x = x.view(paint.shape)
    return x

    return paint


def dilate_mask(mask,opt):
    if opt.mode == "harmonization":
        element = morphology.disk(radius=7)
    if opt.mode == "editing":
        element = morphology.disk(radius=20)
    mask = torch2uint8(mask)
    mask = mask[:, :, 0]
    mask = morphology.binary_dilation(mask, selem=element)
    mask = filters.gaussian(mask, sigma=5)
    nc_im = opt.nc_im
    opt.nc_im = 1
    mask = np2torch(mask, opt)
    opt.nc_im = nc_im
    mask = mask.expand(1, 3, mask.shape[2], mask.shape[3])
    plt.imsave('%s/%s_mask_dilated.png' % (opt.ref_dir, opt.ref_name[:-4]), convert_image_np(mask), vmin=0, vmax=1)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask
