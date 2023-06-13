import cv2
import os
import argparse
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from models import DnCNN, ResCNN, FFTResCNN
from utils import *

# ===================================================================
# This file is used to print noisy images and the denoised images given a certain model.

# TODO: Command line example:
# python3 save_image.py --cnn_model fftrescnn --num_of_layers 18 --num_of_resblocks 6 --logdir logs --test_img 01.png --test_noiseL 25
# ===================================================================


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--cnn_model", type=str, default="fftrescnn", help="The model is one of dncnn, rescnn, fftrescnn")
parser.add_argument("--num_of_layers", type=int, default=24, help="Number of total layers")
parser.add_argument("--num_of_resblocks", type=int, default=8, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_folder", type=str, default='data/Set12/', help='test image folder')
parser.add_argument("--test_img", type=str, default='01', help='test image name')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test image')
opt = parser.parse_args()


def selected_model():
    """
    return the model to train given by the command,
    raise ValueError is the type of model not found
    """
    cnn_chosen = None
    # Here are three models to alternate among
    if opt.cnn_model == 'dncnn':
        cnn_chosen = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    elif opt.cnn_model == 'rescnn':
        cnn_chosen = ResCNN(channels=1, num_of_layers=opt.num_of_layers, num_of_resblocks=opt.num_of_resblocks)
    elif opt.cnn_model == 'fftrescnn':
        cnn_chosen = FFTResCNN(channels=1, num_of_layers=opt.num_of_layers, num_of_resblocks=opt.num_of_resblocks)
    else:
        ValueError("Type of cnn_model is one of: dncnn, rescnn, fftrescnn")
    return cnn_chosen


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("mps")

    # Build model
    print('Loading model ...\n')
    net = selected_model()
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    test_folder = opt.test_folder
    test_img = opt.test_img

    # prepare the image
    image = cv2.imread(test_folder + test_img)
    img_size = image.shape[0]
    image = (np.float32(image[:,:,0]))/255.
    image = np.expand_dims(image, 0)
    image = np.expand_dims(image, 1)
    image_gt = torch.Tensor(image)
    # add the Gaussian noise to get the noisy image
    noise = torch.FloatTensor(image_gt.size()).normal_(mean=0, std=opt.test_noiseL/255.)
    image_noisy = image_gt + noise
    image_gt, image_noisy = Variable(image_gt.to(device)), Variable(image_noisy.to(device))
    with torch.no_grad(): # this can save much memory
        output = torch.clamp(image_noisy-model(image_noisy), 0., 1.)
    psnr_noisy = batch_PSNR(image_noisy, image_gt, 1.)
    psnr_denoised = batch_PSNR(output, image_gt, 1.)
    print("\nPSNR on test image: %f" % psnr_denoised)

    image_gt = image_gt.reshape(img_size, img_size, -1).cpu()
    image_noisy = image_noisy.reshape(img_size, img_size, -1).cpu()
    output = output.reshape(img_size, img_size, -1).cpu()

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 30))
    axs[0].imshow(image_gt, cmap='gray')
    axs[0].axis('off')
    axs[0].set_title("ground truth")
    axs[0].title.set_size(10)
    axs[1].imshow(image_noisy, cmap='gray')
    axs[1].axis('off')
    axs[1].set_title(f"noisy image, PSNR = {psnr_noisy:.2f}")
    axs[1].title.set_size(10)
    axs[2].imshow(output, cmap='gray')
    axs[2].axis('off')
    axs[2].set_title(f"denoised image, PSNR = {psnr_denoised:.2f}")
    axs[2].title.set_size(10)
    plt.savefig(f"dnCNN_{test_img}", bbox_inches='tight')


if __name__ == "__main__":
    main()
