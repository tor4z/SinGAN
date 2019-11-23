from .utils.config import get_arguments, post_config
from summary import Summary
from saver import Saver
from .models.sin_gan import SinGAN
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    opt = post_config(opt)

    summary =  Summary(opt)
    saver = Saver(opt)

    sin_gan = SinGAN(opt, summary, saver)
    sin_gan.generate(Gs, Zs, reals, NoiseAmp, opt)
