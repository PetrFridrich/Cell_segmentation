import matplotlib.pyplot as plt
import mahotas as mh
import numpy as np

import cv2

from PIL import ImageFilter, Image
from scipy import ndimage


def segmentation_process(img,output_path):

    #  Uložení původního snímku
    plt.imsave(f'{output_path}IMG/00_Input_data.png', img)

    #  ------------------ Staticky nastavený proměnný ---------------------

    W = [0.05,0.9,0.05]
    sigma = 10
    mask = np.ones((3,3),np.uint8)
    iterations = 3
    min_size = 200

    #  ------------------ Zde je proces segmentace ------------------------

    #  Zaostření snímku
    img_sharpened = img_sharpening(img)
    plt.imsave(f'{output_path}IMG/01_Sharpened.png', img_sharpened)

    #  Převod snímku na stupně šedi
    img_grayscale = convert_RGB_to_grayscale(img_sharpened, W)
    plt.imsave(f'{output_path}IMG/02_Grayscale.png', img_grayscale, cmap='gray')

    #  Prahování stupňů šedi
    img_bin = convert_grayscale_to_bin(img_grayscale, less_than=False)
    plt.imsave(f'{output_path}IMG/03_Binary.png', img_bin, cmap='gray')

    #  Eroze binárního snímku
    img_bin = cv2.erode(img_bin, mask, iterations = iterations)
    plt.imsave(f'{output_path}IMG/04_Binary_erode.png', img_bin, cmap='gray')

    #  Dilatace binárního snímku
    img_bin = cv2.dilate(img_bin, mask, iterations=iterations)
    plt.imsave(f'{output_path}IMG/05_Binary_dilate.png', img_bin, cmap='gray')

    #  Odstranení malých regionů
    img_bin = remove_small_regions(img_bin, min_size=min_size)
    plt.imsave(f'{output_path}IMG/06_Binary_removed_small_regions.png', img_bin, cmap='gray')

    #  Gaussův filter na stupně šedi
    img_grayscale_gauss = mh.stretch(mh.gaussian_filter(img_grayscale.astype(float), sigma))
    plt.imsave(f'{output_path}IMG/07_Grayscale_gauss_filter.png', img_grayscale_gauss, cmap='gray')

    #  Hledání regionálních maxim v stretchnutém snímku
    img_regional_max = mh.regmax(img_grayscale_gauss)
    plt.imsave(f'{output_path}IMG/08_Binary_regional_maxima.png', img_regional_max, cmap='gray')

    #  Označení jednotlivých regionálních maxim
    img_regional_max_labeled, _ = mh.label(img_regional_max)
    plt.imsave(f'{output_path}IMG/09_Labeled_regional_maxima.png', img_regional_max_labeled, cmap='jet')

    #  Vzdálenostní tranformace v binárních snímku reverznutá a stretchnutá
    img_distance_transform = 255 - mh.stretch(mh.distance(img_bin))
    plt.imsave(f'{output_path}IMG/10_Distance_transform.png', img_distance_transform, cmap='gray')

    #  Aplikace watershedu
    img_watershed = mh.cwatershed(img_distance_transform, img_regional_max_labeled)
    plt.imsave(f'{output_path}IMG/11_Watershed.png', img_watershed, cmap='jet')

    #  Watershed v binární masce
    img_watershed = img_watershed * img_bin
    plt.imsave(f'{output_path}IMG/12_Watershed_mask.png', img_watershed, cmap='jet')

    #  Watershed bez malých regionů
    img_watershed = mh.labeled.remove_regions_where(img_watershed,mh.labeled.labeled_size(img_watershed) < min_size)
    plt.imsave(f'{output_path}IMG/13_Watershed_removed_small_regions.png', img_watershed, cmap='jet')

    #  Watershed relabeled
    img_watershed, _ = mh.labeled.relabel(img_watershed)
    plt.imsave(f'{output_path}IMG/14_Watershed_relabeled.png', img_watershed, cmap='jet')

    #  Watershed cell correction
    img_watershed_1 = cell_correction(img_watershed)
    plt.imsave(f'{output_path}IMG/15_Watershed_corrected.png', img_watershed_1, cmap='jet')

    #  Watershed odstranění hraničních buněk
    img_watershed_2 = mh.labeled.remove_bordering(img_watershed_1)
    plt.imsave(f'{output_path}IMG/16_Watershed_bordering.png', img_watershed_2, cmap='jet')

    #  Watershed relabeled
    img_watershed_2, _ = mh.labeled.relabel(img_watershed_2)
    plt.imsave(f'{output_path}IMG/17_Watershed_bordering_relabeled.png', img_watershed_2, cmap='jet')

    return img_watershed_1, img_watershed_2


def img_sharpening(img):

    RADIUS = 10
    PERCENT = 300
    THRESHOLD = 3

    img_pil = Image.fromarray(img, 'RGB')

    bmp = img_pil.filter(ImageFilter.UnsharpMask(radius = RADIUS, percent = PERCENT, threshold = THRESHOLD))

    return np.array(bmp)


def convert_RGB_to_grayscale(img, W=None):

    if W is None:
        W = [1 / 3, 1 / 3, 1 / 3]

    img_grayscale = img[:,:,0]*W[0] + img[:,:,1]*W[1] + img[:,:,2]*W[2]

    return img_grayscale


def convert_grayscale_to_bin(img, threshold_value = None,less_than = True):

    if threshold_value is None: threshold_value = img.mean()

    if less_than:
        img_bin = img < threshold_value
    else:
        img_bin = img > threshold_value

    return img_bin.astype(np.uint8)


def convert_labeled_to_bin(img, background = 0):

    img_bin = img != background

    return img_bin


def remove_small_regions(img, min_size = 200):

    img,_ = mh.label(img)

    sizes = mh.labeled.labeled_size(img)

    img_without_small_regions = mh.labeled.remove_regions_where(img,sizes < min_size)

    return convert_labeled_to_bin(img_without_small_regions)


def cell_correction(img):

    #  Počet buněk
    n = np.amax(img)

    #  Výsledný snímek
    img_corrected = np.zeros_like(img)

    #  Průchod přes všechny buňky
    for i in range(1,n+1):

        #  Jedna buňka ve snímku
        img_single = img == i

        #  Buňka je dilatována (oprava děr) a následně erodována (vrácena na původní velikost)
        img_single = ndimage.grey_dilation(img_single, size=(9, 9), mode='wrap')
        img_single = ndimage.grey_erosion(img_single, size=(9, 9), mode='wrap')

        #  Přiřazení původního identifikátoru
        img_single = img_single * i

        #  Vložení opravené buňky do výsledného snímku
        img_corrected = img_corrected + img_single

        #  Řešení možného překryvu tak nastavím na identifikátor buňky
        img_corrected[img_corrected > i] = i

    return img_corrected


if __name__ == "__main__":

    print('Hello, home!')