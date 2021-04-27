from itertools import combinations_with_replacement
from collections import defaultdict

import numpy as np
from numpy.linalg import inv

"""
Adjusted python code from https://github.com/joyeecheung/dark-channel-prior-dehazing 
"""


def boxfilter(img, r):
    width, height = img.shape
    dst = np.zeros(img.shape)

    # cumulative sum over Y axis
    sum_y = np.cumsum(img, axis=0)
    # difference over Y axis
    dst[:r + 1] = sum_y[r: 2 * r + 1]
    dst[r + 1:width - r] = sum_y[2 * r + 1:] - sum_y[:width - 2 * r - 1]
    dst[-r:] = np.tile(sum_y[-1], (r, 1)) - sum_y[width - 2 * r - 1:width - r - 1]

    # cumulative sum over X axis
    sum_y = np.cumsum(dst, axis=1)
    # difference over Y axis
    dst[:, :r + 1] = sum_y[:, r:2 * r + 1]
    dst[:, r + 1:height - r] = sum_y[:, 2 * r + 1:] - sum_y[:, :height - 2 * r - 1]
    dst[:, -r:] = np.tile(sum_y[:, -1][:, None], (1, r)) - sum_y[:, height - 2 * r - 1:height - r - 1]

    return dst


def filter(guide, src, r=40, eps=1e-3):
    R = 0
    G = 1
    B = 2

    width, height = src.shape
    base = boxfilter(np.ones((width, height)), r)

    # each channel of I filtered with the mean filter
    means = [boxfilter(guide[:, :, i], r) / base for i in range(3)]
    # p filtered with the mean filter
    mean_p = boxfilter(src, r) / base
    # filter I with p then filter it with the mean filter
    means_IP = [boxfilter(guide[:, :, i] * src, r) / base for i in range(3)]
    # covariance of (I, p) in each local patch
    covIP = [means_IP[i] - means[i] * mean_p for i in range(3)]

    # variance of I in each local patch: the matrix Sigma in ECCV10 eq.14
    var = defaultdict(dict)
    for i, j in combinations_with_replacement(range(3), 2):
        var[i][j] = boxfilter(
            guide[:, :, i] * guide[:, :, j], r) / base - means[i] * means[j]

    a = np.zeros((width, height, 3))
    for y, x in np.ndindex(width, height):
        #         rr, rg, rb
        # Sigma = rg, gg, gb
        #         rb, gb, bb
        Sigma = np.array([[var[R][R][y, x], var[R][G][y, x], var[R][B][y, x]],
                          [var[R][G][y, x], var[G][G][y, x], var[G][B][y, x]],
                          [var[R][B][y, x], var[G][B][y, x], var[B][B][y, x]]])
        cov = np.array([c[y, x] for c in covIP])
        a[y, x] = np.dot(cov, inv(Sigma + eps * np.eye(3)))  # eq 14

    # ECCV10 eq.15
    b = mean_p - a[:, :, R] * means[R] - \
        a[:, :, G] * means[G] - a[:, :, B] * means[B]

    # ECCV10 eq.16
    q = (boxfilter(a[:, :, R], r) * guide[:, :, R] + boxfilter(a[:, :, G], r) *
         guide[:, :, G] + boxfilter(a[:, :, B], r) * guide[:, :, B] + boxfilter(b, r)) / base

    return q
