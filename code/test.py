import cv2
import os
import argparse
import glob
import torch
from torch.autograd import Variable
from models import DnCNN, ResCNN, FFTResCNN
from utils import *

# ===================================================================
# This file is used to test the model. The parameters might be need to be
# specified are shown at the beginning. The cnn_model, num_of_layers, and
# num_of_resblocks must correspond to the training model. There are
# two dataset and three noise levels to test

# TODO: Command line example:
# python3 test.py --cnn_model fftrescnn --num_of_layers 18 --num_of_resblocks 6 --logdir logs --test_data Set12 --test_noiseL 15
# ===================================================================


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="FFTResCNN_Test")
parser.add_argument("--cnn_model", type=str, default="fftrescnn", help="The model is one of dncnn, rescnn, fftrescnn")
parser.add_argument("--num_of_layers", type=int, default=18, help="Number of total layers")
parser.add_argument("--num_of_resblocks", type=int, default=6, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
opt = parser.parse_args()


def selected_model():
    """
        return the model to train given by the command,
        raise ValueError is the type of model not found
    """
    test_model = None
    # Here are three models to alternate among
    if opt.cnn_model == 'dncnn':
        test_model = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    elif opt.cnn_model == 'rescnn':
        test_model = ResCNN(channels=1, num_of_layers=opt.num_of_layers, num_of_resblocks=opt.num_of_resblocks)
    elif opt.cnn_model == 'fftrescnn':
        test_model = FFTResCNN(channels=1, num_of_layers=opt.num_of_layers, num_of_resblocks=opt.num_of_resblocks)
    else:
        ValueError("Type of cnn_model is one of: dncnn, rescnn, fftrescnn")
    return test_model


def compute_psnr(curr_psnr):
    """
    apply each test data to the model and calculate the PSNR value
    """
    # prepare the image
    image = cv2.imread(file)
    image = np.float32(image[:, :, 0])/255.
    image = np.expand_dims(image, 0)
    image = np.expand_dims(image, 1)
    image_gt = torch.Tensor(image)
    # add the Gaussian noise to get the noisy image
    noise = torch.FloatTensor(image_gt.size()).normal_(mean=0, std=opt.test_noiseL / 255.)
    image_noisy = image_gt + noise
    image_gt, image_noisy = Variable(image_gt.cuda()), Variable(image_noisy.cuda())
    with torch.no_grad():
        Out = torch.clamp(image_noisy - model(image_noisy), 0., 1.)
    psnr = batch_PSNR(Out, image_gt, 1.)
    curr_psnr += psnr
    print("%s PSNR %f" % (file, psnr))
    return curr_psnr


if __name__ == "__main__":
    # Load model
    print('Loading model ...\n')
    net = selected_model()
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()
    # process data
    psnr_test = 0
    for file in files_source:
        # apply each test data to the model and calculate the PSNR value
        psnr_test = compute_psnr(psnr_test)
    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)
