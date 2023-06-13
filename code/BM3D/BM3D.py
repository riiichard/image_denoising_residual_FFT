# implementing BM3D based on
# Image denoising by sparse 3D transform-domain collaborative filtering
# https://webpages.tuni.fi/foi/GCF-BM3D/BM3D_TIP_2007.pdf
# and http://www.ipol.im/pub/art/2012/l-bm3d/
# An Analysis and Implementation of the BM3D Image Denoising Method
import cv2
import os
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

# hyper-parameters, according to http://www.ipol.im/pub/art/2012/l-bm3d/
kaiser_window_b = 2.0
sigma = 25
hardThreshold_3d = 2.7 * sigma

firstMatchThreshold = 2500  # threshold for box matching
s1_n_match = 16  # max number of boxes in each matched group
s1_block_size = 8  # block size
s1_block_step = 3  # loop over the pixels of the image with step p (integer) in row and column
s1_search_step = 3  # number of search step for block
s1_search_window = 39  # search window size

secondMatchThreshold = 400
s2_n_match = 32
s2_block_size = 8
s2_block_step = 3
s2_search_step = 3
s2_search_window = 39

# ref to https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

# get the initialize image weight and kaiser window
def weighting_and_kaiser(image, _block_size, _kaiser_window_b):
    shape = image.shape
    img = np.matrix(np.zeros(shape, dtype=float))
    weight = np.matrix(np.zeros(shape, dtype=float))
    K = np.matrix(np.kaiser(_block_size, _kaiser_window_b))
    Kaiser = np.array(K.T * K)
    return img, weight, Kaiser


# ensure that the current block does not exceed the image range
def create_block(i, j, block_step, block_size, w, h):
    if block_size + block_step * i >= w:
        x = w - block_size
    else:
        x = i * block_step
    if block_size + block_step * j >= h:
        y = h - block_size
    else:
        y = j * block_step
    block = np.array((x, y), dtype=int)
    return block


# (x,y) tuple defines vertex coordinates of the search window.
def search_window_loc(_noisy, _block, _window_size, block_size):
    (x, y) = _block

    # 4 points coordinates
    x1 = x + block_size / 2 - _window_size / 2
    y1 = y + block_size / 2 - _window_size / 2
    x2 = x1 + _window_size
    y2 = y1 + _window_size

    # if exceeds, set to the edge
    if x1 < 0:
        x1 = 0
    elif x2 > _noisy.shape[0]:
        x1 = _noisy.shape[0] - _window_size
    if y1 < 0:
        y1 = 0
    elif y2 > _noisy.shape[0]:
        y1 = _noisy.shape[0] - _window_size

    coord = np.array((x1, y1), dtype=int)
    return coord


# return the blocks with the highest similarity to the current block around
def s1_block_match(_noisy, _block):
    (x_current, y_current) = _block

    # parameter
    threshold = firstMatchThreshold
    max_match = s1_n_match
    block_size = s1_block_size
    search_step = s1_search_step
    window_size = s1_search_window

    # Discrete Cosine Transform Init
    img = _noisy[x_current: x_current + block_size, y_current: y_current + block_size]
    img_dct = cv2.dct(img.astype(np.float64))
    # the locations of similar blocks
    similar_blocks_group = np.zeros((max_match, block_size, block_size))
    blocks_loc = np.zeros((max_match, 2), dtype=int)
    similar_blocks_group[0, :, :] = img_dct
    blocks_loc[0, :] = _block
    # the location of window
    window_loc = search_window_loc(_noisy, _block, window_size, block_size)
    (x_current, y_current) = window_loc
    # the number of similar blocks
    n_block = int((window_size - block_size) / search_step)
    similar_blocks = np.zeros((n_block ** 2, block_size, block_size))
    block_positions = np.zeros((n_block ** 2, 2), dtype=int)
    # similarity
    d = np.zeros(n_block ** 2)

    # nested for-loop search in the search window, find similar blocks
    n_match = 0
    for i in range(n_block):
        for j in range(n_block):
            temp = _noisy[x_current: x_current + block_size, y_current: y_current + block_size]
            temp_dct = cv2.dct(temp.astype(np.float64))
            distance = np.linalg.norm((img_dct - temp_dct)) ** 2 / (block_size ** 2)
            if 0 < distance < threshold:
                similar_blocks[n_match, :, :] = temp_dct
                block_positions[n_match, :] = (x_current, y_current)
                d[n_match] = distance
                # find new match
                n_match += 1
            y_current += search_step
        x_current += search_step
        y_current = window_loc[1]
    d = d[:n_match]

    # calculate the number of matching blocks
    if n_match >= max_match:
        count = max_match
    else:
        count = n_match + 1
    if count > 0:
        for i in range(1, count):
            similar_blocks_group[i, :, :] = similar_blocks[d.argsort()[i - 1], :, :]
            blocks_loc[i, :] = block_positions[d.argsort()[i - 1], :]

    # return similar blocks group, their locations, and the number of matching blocks
    return similar_blocks_group, blocks_loc, count


# process 3D transform and filtering
def s1_3D_filtering(_similar_blocks, hardThreshold_3d):
    n_nonzero = 0
    for i in range(_similar_blocks.shape[1]):
        for j in range(_similar_blocks.shape[2]):
            blocks_dct = cv2.dct(_similar_blocks[:, i, j])
            blocks_dct[np.abs(blocks_dct[:]) < hardThreshold_3d] = 0.
            n_nonzero += blocks_dct.nonzero()[0].size
            _similar_blocks[:, i, j] = cv2.idct(blocks_dct)[0]
    return _similar_blocks, n_nonzero


# weighted accumulation of the output stacks after 3D transform and filtering, to obtain the first step image
def s1_aggregation(_similar_blocks, block_positions, basic_img, wight_img, n_nonzero, count, Kaiser):
    if n_nonzero < 1:
        n_nonzero = 1
    for i in range(count):
        p = block_positions[i, :]
        temp = (1. / n_nonzero) * cv2.idct(_similar_blocks[i, :, :]) * Kaiser
        basic_img[p[0]:p[0] + _similar_blocks.shape[1], p[1]:p[1] + _similar_blocks.shape[2]] += temp
        wight_img[p[0]:p[0] + _similar_blocks.shape[1], p[1]:p[1] + _similar_blocks.shape[2]] += (1 / n_nonzero) * Kaiser

# return the blocks with the highest similarity to the current block around
def s2_block_match(_img_s1, _noisy, _block):
    (x_current, y_current) = _block

    # parameter
    threshold = secondMatchThreshold
    max_match = s2_n_match
    block_size = s2_block_size
    search_step = s2_search_step
    window_size = s2_search_window

    # Discrete Cosine Transform Init
    img = _img_s1[x_current: x_current + block_size, y_current: y_current + block_size]
    img_dct = cv2.dct(img.astype(np.float32))
    n_img = _noisy[x_current: x_current + block_size, y_current: y_current + block_size]
    n_img_dct = cv2.dct(n_img.astype(np.float32))
    # the locations of similar blocks
    similar_blocks_group = np.zeros((max_match, block_size, block_size))
    noisy_blocks_group = np.zeros((max_match, block_size, block_size))
    block_loc = np.zeros((max_match, 2), dtype=int)
    similar_blocks_group[0, :, :] = img_dct
    noisy_blocks_group[0, :, :] = n_img_dct
    block_loc[0, :] = _block
    # the location of window
    window_loc = search_window_loc(_noisy, _block, window_size, block_size)
    (x_current, y_current) = window_loc
    # the number of similar blocks
    n_block = int((window_size - block_size) / search_step)
    similar_blocks = np.zeros((n_block ** 2, block_size, block_size))
    block_positions = np.zeros((n_block ** 2, 2), dtype=int)
    # similarity
    d = np.zeros(n_block ** 2)

    # nested for-loop search in the search window, find similar blocks
    n_match = 0
    for i in range(n_block):
        for j in range(n_block):
            temp = _img_s1[x_current: x_current + block_size, y_current: y_current + block_size]
            temp_dct = cv2.dct(temp.astype(np.float32))
            distance = np.linalg.norm((img_dct - temp_dct)) ** 2 / (block_size ** 2)
            if threshold > distance > 0:
                similar_blocks[n_match, :, :] = temp_dct
                block_positions[n_match, :] = (x_current, y_current)
                d[n_match] = distance
                # find new match
                n_match += 1
            y_current += search_step
        x_current += search_step
        y_current = window_loc[1]
    d = d[:n_match]

    # calculate the number of matching blocks
    if n_match >= max_match:
        count = max_match
    else:
        count = n_match + 1
    if count > 0:
        for i in range(1, count):
            similar_blocks_group[i, :, :] = similar_blocks[d.argsort()[i - 1], :, :]
            block_loc[i, :] = block_positions[d.argsort()[i - 1], :]
            (x_current, y_current) = block_positions[d.argsort()[i - 1], :]
            n_img = _noisy[x_current: x_current + block_size, y_current: y_current + block_size]
            noisy_blocks_group[i, :, :] = cv2.dct(n_img.astype(np.float64))

    return similar_blocks_group, noisy_blocks_group, block_loc, count


# process 3D transform and filtering
def s2_3D_filtering(_similar_blocks, _similar_imgs):
    wiener_wight = np.zeros((_similar_blocks.shape[1], _similar_blocks.shape[2]))
    for i in range(_similar_blocks.shape[1]):
        for j in range(_similar_blocks.shape[2]):
            blocks_dct = np.matrix(cv2.dct(_similar_blocks[:, i, j]))
            weight = np.float64(blocks_dct.T * blocks_dct) / (np.float64(blocks_dct.T * blocks_dct) + sigma ** 2)
            if weight != 0:
                wiener_wight[i, j] = 1 / (weight ** 2 * sigma ** 2)
            blocks_dct = weight * cv2.dct(_similar_imgs[:, i, j])
            _similar_blocks[:, i, j] = cv2.idct(blocks_dct)[0]

    return _similar_blocks, wiener_wight

# weighted accumulation of the output stacks after 3D transform and filtering, to obtain the final image
def s2_aggregation(_similar_blocks, block_positions, basic_img, wight_img, _wiener_wight, count):
    for i in range(count):
        p = block_positions[i, :]
        temp = _wiener_wight * cv2.idct(_similar_blocks[i, :, :])
        basic_img[p[0]:p[0] + _similar_blocks.shape[1], p[1]:p[1] + _similar_blocks.shape[2]] += temp
        wight_img[p[0]:p[0] + _similar_blocks.shape[1], p[1]:p[1] + _similar_blocks.shape[2]] += _wiener_wight

# first step
def step1(_noisy):
    # parameter
    block_size = s1_block_size
    block_step = s1_block_step
    (w, h) = _noisy.shape
    n_w = (w - block_size) / block_step
    n_h = (h - block_size) / block_step

    # get the initialize image weight and kaiser window
    img_s1, weight, Kaiser = weighting_and_kaiser(_noisy, block_size, kaiser_window_b)

    # nested loop for each block, plus 2 to avoid insufficient on the edge
    for i in range(int(n_w + 2)):
        for j in range(int(n_h + 2)):
            # ensure that the current block does not exceed the image range
            current_block = create_block(i, j, block_step, block_size, w, h)
            # block match
            similar_blocks, blocks_loc, count = s1_block_match(_noisy, current_block)
            # process 3D transform and filtering(hard thresholding)
            similar_blocks, n_nonzero = s1_3D_filtering(similar_blocks, hardThreshold_3d)
            # aggregation
            s1_aggregation(similar_blocks, blocks_loc, img_s1, weight, n_nonzero, count, Kaiser)

    img_s1[:, :] /= weight[:, :]
    img_s1 = np.matrix(img_s1, dtype=int).astype(np.uint8)

    return img_s1

#second step
def step2(_img_s1, _noisy):
    # parameter
    block_size = s2_block_size
    block_step = s2_block_step
    (w, h) = _noisy.shape
    n_w = (w - block_size) / block_step
    n_h = (h - block_size) / block_step

    # get the initialize image weight and kaiser window
    img_s2, weight, Kaiser = weighting_and_kaiser(_noisy, block_size, kaiser_window_b)

    # nested loop for each block, plus 2 to avoid insufficient on the edge
    for i in range(int(n_w + 2)):
        for j in range(int(n_h + 2)):
            # ensure that the current block does not exceed the image range
            m_blockPoint = create_block(i, j, block_step, block_size, w, h)
            # block match
            similar_blocks, similar_imgs, blocks_loc, count = s2_block_match(_img_s1, _noisy, m_blockPoint)
            # process 3D transform and filtering(hard thresholding)
            similar_blocks, wiener_wight = s2_3D_filtering(similar_blocks, similar_imgs)
            # aggregation
            s2_aggregation(similar_blocks, blocks_loc, img_s2, weight, wiener_wight, count)
    img_s2[:, :] /= weight[:, :]
    final_img = np.matrix(img_s2, dtype=int)
    final_img.astype(np.uint8)

    return final_img


def normalize(data):
    return data / 255.


def evaluate_model(dataset, sigma=10, output_filename_1='out1.png', output_filename_2='out2.png'):
    psnrs_first = []
    psnrs_second = []
    for idx, image in enumerate(dataset):
        # add noise to original image
        # normalize
        Img = normalize(image)
        # noise
        noise = np.random.normal(0, sigma / 255, (Img.shape[0], Img.shape[1]))
        # noisy image
        INoisy = Img + noise
        # BM3D-first step
        img_s1 = step1(INoisy * 255)
        # calculate PSNR
        psnr_first = peak_signal_noise_ratio(image, img_s1)
        psnrs_first.append(psnr_first)
        # BM3D-second step
        final_img = step2(img_s1, INoisy * 255)
        # calculate PSNR
        psnr_second = peak_signal_noise_ratio(image, final_img)
        psnrs_second.append(psnr_second)

        return np.mean(psnrs_first), np.mean(psnrs_second)



if __name__ == '__main__':
    cv2.setUseOptimized(True)

    # load image
    set12 = load_images_from_folder('data/Set12')
    set68 = load_images_from_folder('data/Set68')

    sigmas = [15, 25, 50]

    # set performance, obtain average psnr
    for sigma in sigmas:
        psnr1, psnr2 = evaluate_model(set12, sigma=sigma, output_filename_1=f'set12_sigma_{sigma}_first.png',
                                      output_filename_2=f'set12_sigma_{sigma}_second.png')
        print(f'\tSet 12 Results with sigma = {sigma}: {psnr1:.04f} dB | {psnr2:.04f} dB')

    for sigma in sigmas:
        psnr1, psnr2 = evaluate_model(set68, sigma=sigma, output_filename_1=f'set68_sigma_{sigma}_first.png',
                                      output_filename_2=f'set68_sigma_{sigma}_second.png')
        print(f'\tSet 68 Results with sigma = {sigma}: {psnr1:.04f} dB | {psnr2:.04f} dB')

    # for testing images result
    test = load_images_from_folder('data/test')
    index = 0
    for idx, image in enumerate(test):
        index += 1
        img = normalize(image)
        noiseimgs = {}
        psnrs = []
        denoise = {}
        for sigma in sigmas:
            # noise
            noise = np.random.normal(0, sigma / 255, (img.shape[0], img.shape[1]))
            # noisy image
            INoisy = img + noise
            # cv2.imwrite(f'image with noise_{sigma}_2.png', INoisy)
            noiseimgs[f'image with noise_{sigma}'] = INoisy
            # BM3D-first step
            Basic_img = step1(INoisy * 255)
            # BM3D-second step
            Final_img = step2(Basic_img, INoisy * 255)
            denoise[f'denoise image s2_{sigma}'] = Final_img
            # calculate PSNR
            psnr = peak_signal_noise_ratio(image, Final_img)
            psnrs.append(psnr)

            #save single result
            skimage.io.imsave(f'BM3D_{index}_sigma_{sigma}_psnr_{psnr}.png', Final_img)


        fig, ax = plt.subplots(nrows=2, ncols=3,figsize=(30,30))
        for r, sigma in enumerate(sigmas):
            ax[0,r].imshow(noiseimgs[f'image with noise_{sigma}'], cmap='gray', vmin=0, vmax=1)
            ax[0,r].set_title(f'Noisy Image, sigma={sigma}')
            ax[0,r].axis('off')

            ax[1,r].imshow(denoise[f'denoise image s2_{sigma}'], cmap='gray', vmin=0, vmax=255)
            ax[1,r].set_title(f'BM3D Denoising Image, PSNR={np.around(psnrs[r],4)}dB')
            ax[1,r].axis('off')

        fig.savefig(f'BM3D_{index}.png', bbox_inches='tight')
