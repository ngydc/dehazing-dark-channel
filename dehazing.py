import numpy as np
import math


def dark_channel(image, window):
    """
    computes the dark channel from a given image (EQ 5)
    :param image: image as a numpy 2D array
    :param window: size of the window
    :return: the dark channel of the given image
    """
    m, n, _ = image.shape
    pad = int(math.floor(window / 2))
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), 'edge')
    dark_ch = np.zeros((m, n))
    for i, j in np.ndindex(dark_ch.shape):
        dark_ch[i, j] = np.min(padded[i:i + window, j:j + window, :])
    return dark_ch


def atmospheric_light(image, dark_ch, p=0.01):
    """
    computes the atmospheric light from a given image (SEC 4.3)
    :param image: image as a numpy 2D array
    :param dark_ch: dark channel of the given image
    :param p: probability of the brightest pixels
    :return: the computed atmospheric light
    """
    m, n = dark_ch.shape
    flat_image = np.reshape(image, (m * n, 3))
    flat_dark = dark_ch.flatten()

    search_idx = np.argsort(-flat_dark)[:int(m * n * p)]
    return np.max(flat_image.take(search_idx, axis=0), axis=0)


def transmission(image, atm_light, window, omega=0.95):
    """
    computes the transmission from a given image (EQ 12)
    :param image: image as a numpy 2D array
    :param atm_light: atmospheric light of the image
    :param omega: value which affects the amount of haze which is kept
    :param window: window size
    :return: the estimated transmission
    """
    normalize = np.zeros(image.shape)
    for ind in range(0, 3):
        normalize[:, :, ind] = image[:, :, ind] / atm_light[ind]

    return 1 - omega * dark_channel(normalize, window)


def recover_radiance(image, atm_light, transmission, t0=0.1):
    """
    recovers the scene radiance (dehazed image) from a hazy input image (EQ 22)
    :param image: image as a numpy 2D array
    :param atm_light: atmospheric light of the image
    :param transmission: transmission of the image
    :param t0: lower bound of the transmission
    :return: the scene radiance
    """

    bounded_transmission = np.zeros(image.shape)
    temp = np.copy(transmission)
    temp[temp < t0] = t0

    # convert to shape of the image for easier numpy computation
    for i in range(3):
        bounded_transmission[:, :, i] = temp

    return ((image - atm_light) / bounded_transmission) + atm_light
