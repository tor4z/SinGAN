from utils.config import Config
from utils.summary import Summary
from utils.saver import Saver
from models.sin_gan import SinGAN


def main(opt):
    summary =  Summary(opt)
    saver = Saver(opt)
    sin_gan = SinGAN(opt, summary, saver)

    sin_gan.train_pyramid()
    sin_gan.generate()
    print('Finished.')


if __name__ == '__main__':
    opt = Config('cfg.yaml')
    main(opt)
