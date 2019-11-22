from config import get_arguments
from summary import Summary
from saver import Saver
from SinGAN.manipulate import SinGAN_generate
from SinGAN.training import train
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
    summary =  Summary(opt)
    saver = Saver(opt)

    real = functions.read_image(opt)
    functions.adjust_scales2image(real, opt)
    train(opt, Gs, Zs, reals, NoiseAmp, summary, saver)
    SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt)
