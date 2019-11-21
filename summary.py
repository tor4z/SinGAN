import os
import datetime
import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from SinGAN.functions import denorm


class Summary:
    def __init__(self, opt):
        date_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        train_obj = opt.input_name.split('.')[0]
        path = os.path.join(opt.log_dir, train_obj, opt.mode, date_str)
        self.writer = SummaryWriter(log_dir=path)
        self.opt = opt

    def tensor_to_single_image(self, tensor):
        tensor = denorm(tensor.clone().detach().cpu())
        if tensor.shape[1] == 3:
            return tensor[-1, :, :, :]
        else:
            return tensor[-1, -1, :, :]

    def tensor_to_multi_image(self, tensor):
        tensor = denorm(tensor.clone().detach().cpu())
        if tensor.shape[1] == 3:
            count = min(opt.imgs_write, tensor.shape[0])
            return tensor[-count:, :, :, :]
        else:
            count = min(opt.imgs_write, tensor.shape[1])
            return tensor[-1:, -count, :, :]

    def add_image(self, tag, img_tensor, global_steps, *agrs, **kwargs):
        img_tensor = self.tensor_to_single_image(img_tensor)
        self.writer.add_image(tag, img_tensor, global_steps, *agrs, **kwargs)

    def add_images(self, tag, img_tensor, global_steps, *agrs, **kwargs):
        img_tensor = self.tensor_to_multi_image(img_tensor)
        self.writer.add_images(tag, img_tensor, global_steps, *agrs, **kwargs)

    def add_graph(self, model, *args, **kwargs):
        self.writer.add_graph(model, *args, **kwargs)

    def add_scalar(self, tag, scalar_value, global_steps, *args, **kwargs):
        self.writer.add_scalar(tag, scalar_value, global_steps, *agrs, **kwargs)

    def add_scalars(self, main_tag, tag_scalar_dict, global_steps, *agrs, **kwargs):
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_steps, *agrs, **kwargs)

    def close(self):
        try:
            self.writer.close()
        except:
            pass

    def __del__(self):
        self.close()