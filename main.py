# This is a sample Python script.
import PIL.Image
import pandas as pd
import csv
import math
import numba
import numpy as np
import matplotlib.pyplot as plt
from PIL.Image import Palette
from PIL.ImagePalette import ImagePalette
from skimage import measure as sm

@numba.njit(cache=True)
def downsample_image(a, r, s):
    b, ks = np.zeros_like(a), np.zeros((2 * s + 1, 2 * s + 1), dtype=np.float64)
    for p in range(-s, s + 1):
        for q in range(-s, s + 1):
            ks[p + s, q + s] = (1. - min(s, math.sqrt(p ** 2 + q ** 2)) / s)
    ks /= np.sum(ks)
    for ch in range(a.shape[2]):
        for i in range(s, a.shape[0] - s):
            for j in range(s, a.shape[1] - s):
                c = 0
                for p in range(-s, s + 1):
                    for q in range(-s, s + 1):
                        c += a[i + p, j + q, ch] * ks[p + s, q + s]
                b[i, j, ch] = round(c) // r * r
    return b


def downsample_two(image, mode: str, palette: ImagePalette):
    img_array = np.asarray(image.convert('RGB'), dtype='int32')
    downsample = 6   # Love, you can change it here <3
    ds_array = img_array / 255
    r = sm.block_reduce(ds_array[:, :, 0],
                                     (downsample, downsample),
                                     np.mean)
    g = sm.block_reduce(ds_array[:, :, 1],
                                     (downsample, downsample),
                                     np.mean)
    b = sm.block_reduce(ds_array[:, :, 2],
                                     (downsample, downsample),
                                     np.mean)
    ds_array_downsized = np.stack((r, g, b), axis=-1)

    return ds_array_downsized

def rgb2hex(a: tuple):
    r, g, b = int(a[0]*255), int(a[1]*255), int(a[2]*255)
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def store_array_to_csv(ds_array, csv_name):
    res = np.apply_along_axis(rgb2hex, 2, ds_array)

    result_df = pd.DataFrame(res[:, :]).stack().rename_axis(['y', 'x']).reset_index(name='color')
    # Add ';' in the end of the DataFrame
    # result_df = result_df.append({'y': ';', 'x': '', 'color': ''}, ignore_index=True)
    result_df.columns = ['y', 'x', 'color']
    # Export the DataFrame as csv file
    result_df.to_csv('result.csv', sep=';', index=False)

if __name__ == '__main__':
    image_path = 'image.png'
    new_image_path = 'image2.png'

    image = PIL.Image.open(image_path)
    print(f"The size of {image_path} is {image.size}.")
    mode = 'P'
    image_reduced_colors = image.convert(mode, palette=Palette.ADAPTIVE, colors=12)
    plt.imshow(image_reduced_colors)
    plt.show()

    ds_img = downsample_two(image_reduced_colors, 'A', image_reduced_colors.palette)
    store_array_to_csv(ds_img, 'result.csv')
    plt.imshow(ds_img)
    plt.show()

    # img2 = PIL.Image.fromarray(downsample_image(np.array(image), 32, 8))

    # print(f"Saving the new converted, downsampled image to {new_image_path}.")
    # img2.save(new_image_path)

