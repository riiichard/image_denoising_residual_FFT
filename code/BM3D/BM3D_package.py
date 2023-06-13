import bm3d
import cv2
import os
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

# ref to https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def normalize(data):
    return data/255.

def evaluate_model(dataset,sigma=10,output_filename_1='out1.png',output_filename_2='out2.png'):
    psnrs = []
    for idx, image in enumerate(dataset):
        # add noise to original image
        #normalize
        Img = normalize(image)
        # noise
        noise = np.random.normal(0, sigma/255, (Img.shape[0],Img.shape[1]))
        # noisy image
        INoisy = Img + noise

        # BM3D-first step
        denoise = bm3d.bm3d(INoisy*255,sigma_psd=0.25)
        # calculate PSNR
        psnr = peak_signal_noise_ratio(image, denoise)
        psnrs.append(psnr)
        # if idx == 0:
        #     skimage.io.imsave(output_filename_1, Basic_img)
        #     skimage.io.imsave(output_filename_2, Final_img)
        return np.mean(psnrs)

if __name__ == '__main__':
    cv2.setUseOptimized(True)

    # load image
    set12 = load_images_from_folder('data/Set12')
    set68 = load_images_from_folder('data/Set68')

    sigmas = [15, 25, 50]

    for sigma in sigmas:

        psnr = evaluate_model(set12, sigma=sigma, output_filename_1 = f'set12_sigma_{sigma}_first.png',
                                      output_filename_2 = f'set12_sigma_{sigma}_second.png')
        print(f'\tSet 12 Results with sigma = {sigma}: {psnr:.04f} dB)')

    for sigma in sigmas:
        psnr = evaluate_model(set68, sigma=sigma, output_filename_1 = f'set68_sigma_{sigma}_first.png',
                                      output_filename_2 = f'set68_sigma_{sigma}_second.png')
        print(f'\tSet 68 Results with sigma = {sigma}: {psnr:.04f} dB)')
