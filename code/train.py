import os
import argparse
import torch
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import DnCNN, ResCNN, FFTResCNN
from dataset import prepare_data, Dataset
from utils import *

# ===================================================================
# This file is used to train the model. The parameters might be need to be
# specified are shown at the beginning. If it's the first time to train,
# no h5py file exists, set the 'preprocess' to True, other False. It would
# save an event file and the model in '/logs'.

# TODO: Command line example:
# python3 train.py --prep True --cnn_model fftrescnn --num_of_layers 18 --num_of_resblocks 6 --lr 1e-3 --val_noiseL 25
# ===================================================================


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parameter involved to train
parser = argparse.ArgumentParser(description="FFTResCNN")
parser.add_argument("--prep", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=64, help="Training batch size")
parser.add_argument("--cnn_model", type=str, default="fftrescnn", help="The model is one of dncnn, rescnn, fftrescnn")
parser.add_argument("--num_of_layers", type=int, default=24, help="Number of total layers")
parser.add_argument("--num_of_resblocks", type=int, default=8, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
opt = parser.parse_args()


def data_loader():
    """
    Load the training and validation datasets
    """
    trainData = Dataset(train=True)
    ValData = Dataset(train=False)
    load_train = DataLoader(dataset=trainData, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    return trainData, ValData, load_train


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


def train_epoch():
    """
    training set for each epoch
    """
    # set up the model
    model.train()
    model.zero_grad()
    optimizer.zero_grad()
    # prepare the images and corresponding noisy images
    image_train = data
    noise = torch.zeros(image_train.size())
    stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
    for n in range(noise.size()[0]):
        sizeN = noise[0, :, :, :].size()
        noise[n, :, :, :] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n] / 255.)
    noisy_train = image_train + noise
    image_train, noisy_train = Variable(image_train.cuda()), Variable(noisy_train.cuda())
    noise = Variable(noise.cuda())
    out_train = model(noisy_train)
    # compute the loss function
    loss = criterion(out_train, noise) / (noisy_train.size()[0] * 2)
    loss.backward()
    optimizer.step()
    model.eval()
    out_train = torch.clamp(noisy_train - model(noisy_train), 0., 1.)
    psnr_train = batch_PSNR(out_train, image_train, 1.)
    print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
          (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))
    if step % 10 == 0:
        # Log the scalar values
        writer.add_scalar('loss', loss.item(), step)
        writer.add_scalar('PSNR on training data', psnr_train, step)
    return image_train, noisy_train


def valid_epoch():
    model.eval()
    # validate
    psnr_val = 0
    for k in range(len(dataset_val)):
        img_val = torch.unsqueeze(dataset_val[k], 0)
        noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL / 255.)
        imgn_val = img_val + noise
        img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)
        out_val = torch.clamp(imgn_val - model(imgn_val), 0., 1.)
        psnr_val += batch_PSNR(out_val, img_val, 1.)
    psnr_val /= len(dataset_val)
    print("\n[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_val))
    writer.add_scalar('PSNR on validation data', psnr_val, epoch)


def save_log():
    """
    save the event details of current training
    """
    out_train = torch.clamp(imgn_train - model(imgn_train), 0., 1.)
    Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
    Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
    Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
    writer.add_image('clean image', Img, epoch)
    writer.add_image('noisy image', Imgn, epoch)
    writer.add_image('reconstructed image', Irecon, epoch)


if __name__ == "__main__":
    # If it has not trained before, no h5py file exists, preprocess shoule be set to TRUE,
    # otherwise FALSE, prepare and load the data from the beginning.
    if opt.prep:
        # patch_size, stride and aug_times are important parameter in data processing,
        # effecting the number of training data
        prepare_data(data_path='data', patch_size=50, stride=30, aug_times=2)
    # Load training and validation dataset that has been prepared
    print('Loading dataset ...\n')
    dataset_train, dataset_val, loader_train = data_loader()
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = selected_model()
    # Initialize weights
    net.apply(weights_init_kaiming)
    # Set loss function
    criterion = nn.MSELoss(size_average=False)
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # Prepare training, prepare to write the logs anf events
    writer = SummaryWriter(opt.outf)
    step = 0
    # blind noise training, random noise level chose from 0 to 55, might be changed in future
    noiseL_B = [0, 55]
    # train for each epoch
    for epoch in range(opt.epochs):
        # decay the learning rate as the epoch increases (after a certain milestone point)
        if epoch < opt.milestone:
            # beginning epochs
            current_lr = opt.lr
        else:
            # ending epochs
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # begin the training for current epoch
        for i, data in enumerate(loader_train, 0):
            # training step
            img_train, imgn_train = train_epoch()
            step += 1
        # after each training epoch, save the model in net.pth file
        # TODO: torch.no_grad() is used when the RAM of GPU node is limited/not enough,
        #  remove if more than 12 G
        with torch.no_grad():
            valid_epoch()
            # log the images
            save_log()
            # save model
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))
