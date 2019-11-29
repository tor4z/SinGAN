import argparse
import random
import datetime
import yaml
import torch


class Config(object):
    def __init__(self, cfg=None):
        if cfg is not None:
            with open(cfg) as file:
                self.cfg = yaml.load(file, Loader=yaml.FullLoader)
            self.post_config()
            self.init_attr()

    def init_attr(self):
        for k, v in self.cfg.items():
            self.__setattr__(k, v)

    def post_config(self):
        # init fixed parameters
        if self.cfg['devices']:
            self.cfg['cuda'] = True
            self.cfg['device'] = torch.device('cuda:{}'.format(self.cfg['devices'][0]))
        else:
            self.cfg['cuda'] = False
            self.cfg['device'] = torch.device('cpu')

        self.cfg['niter_init'] = self.cfg['niter']
        self.cfg['noise_amp_init'] = self.cfg['noise_amp']
        self.cfg['nfc_init'] = self.cfg['nfc']
        self.cfg['min_nfc_init'] = self.cfg['min_nfc']
        self.cfg['scale_factor_init'] = self.cfg['scale_factor']

        if self.cfg['mode'] == 'SR':
            self.cfg['alpha'] = 100

        if self.cfg['manualSeed'] is None:
            self.cfg['manualSeed'] = random.randint(1, 10000)
        print("Random Seed: ", self.cfg['manualSeed'])
        random.seed(self.cfg['manualSeed'])
        torch.manual_seed(self.cfg['manualSeed'])

        if self.cfg['resume'] and not self.cfg['runtime_id']:
            raise RuntimeError('when resume is enabled, runtime_id should be specified.')

        if not self.cfg['runtime_id']:
            dtate_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.cfg['runtime_id'] = dtate_str

        self.cfg['device_count'] = len(self.cfg['devices']) or 1

    def dump(self):
        return self.cfg

    def load(self, cfg):
        self.cfg = cfg
        self.init_attr()
