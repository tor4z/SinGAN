import shutil

from config import get_arguments
from summary import Summary
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    opt = functions.post_config(opt)

    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    summary =  Summary(opt)

    if os.path.exists(dir2save):
        if not opt.force:
            print('trained model already exist')
            exit(0)
        else:
            shutil.rmtree(dir2save)

    try:
        os.makedirs(dir2save)
    except OSError:
        print('Can not make dir {}'.format(dir2save))
        exit(1)

    real = functions.read_image(opt)
    functions.adjust_scales2image(real, opt)
    train(opt, Gs, Zs, reals, NoiseAmp, summary)
    SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt)
