import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.fft import fft2, fftshift

# ===================================================================
# This file is used to print the plot used in the Introduction in the report.
# It plots a clean image, its noisy image with some noise level,
# the FFT of the clean image, and the FFT of the noisy image.
# ===================================================================


hw_dir = Path(__file__).parent
# Load images
img = io.imread(hw_dir/'data/Set12/02.png')
img_rgb = img.astype(np.float64)/255

# convert the images into spectrum by channel using FFT
# 2D Fast Fourier Transform
img_spec = fft2(img_rgb)
# shift the spectrum
img_spec = np.abs(fftshift(img_spec))

# add Gaussian noise to the image
noisy = img_rgb + np.random.normal(0, 25/255., (img.shape[0], img.shape[1]))
# 2D Fast Fourier Transform
noisy_spec = fft2(noisy)
# shift the spectrum
noisy_spec = np.abs(fftshift(noisy_spec))

# plot the images
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
axs[0, 0].imshow(img_rgb, cmap='gray')
axs[0, 0].axis('off')
axs[0, 0].set_title("clean image")
axs[0, 0].title.set_size(20)
axs[0, 1].imshow(noisy, cmap='gray')
axs[0, 1].axis('off')
axs[0, 1].set_title("noisy image")
axs[0, 1].title.set_size(20)
axs[1, 0].imshow(np.log(1 + img_spec), cmap='gray')
axs[1, 0].axis('off')
axs[1, 0].set_title("FFT of clean image")
axs[1, 0].title.set_size(20)
axs[1, 1].imshow(np.log(1 + noisy_spec), cmap='gray')
axs[1, 1].axis('off')
axs[1, 1].set_title("FFT of noisy image")
axs[1, 1].title.set_size(20)
plt.savefig("intro.png", bbox_inches='tight')
