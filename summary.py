import os
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter as SW


class SummaryWriter:
    def __init__(self, opt):
        date_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        path = os.path.join(opt.log_dir, opt.input_name, opt.mode, date_str)
        self.writer = SW(log_dir=path)

    def add_image(self, tag, img_tensor, global_steps, *agrs, **kwargs):
        pass

    def add_images(self, tag, img_tensor, global_steps, *agrs, **kwargs):
        pass

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