import numpy as np
import matplotlib.pyplot as plt
import os
import time
import cv2
from shutil import copyfile

PATH = r'media/images/'
CLARITY_PATH = r'media/result/low_quality/'


# edges
def edges(n, orient):
    edges = np.ones((2 * n, 2 * n, 3))

    if orient == 'vert':
        for i in range(0, 2 * n):
            edges[i][n: 2 * n] *= -1
    elif orient == 'horiz':
        edges[n: 2 * n] *= -1

    return edges

# Apply one filter defined by parameters W and single slice
def conv_single_step(a_slice_prev, W):
    s = W * a_slice_prev
    Z = np.sum(s)
    Z = np.abs(Z)

    return Z

# Full edge filter
def conv_forward(A_prev, W, hparameters):
    m = len(A_prev)
    (f, f, n_C) = W.shape
    stride = hparameters['stride']
    pad = hparameters['pad']

    Z = list()
    flag = 0
    z_max = hparameters['z_max']

    if len(z_max) == 0:
        z_max = list()
        flag = 1

    for i in range(m):

        (x0, x1, x2) = A_prev[i].shape
        A_prev_pad = A_prev[i][
                     int(x0 / 3): int(x0 * 2 / 3),
                     int(x1 / 3): int(x1 * 2 / 3),
                     :]

        (n_H_prev, n_W_prev, n_C_prev) = A_prev_pad.shape
        n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
        n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
        z = np.zeros((n_H, n_W))

        a_prev_pad = A_prev_pad

        for h in range(n_H):
            vert_start = h * stride
            vert_end = h * stride + f

            for w in range(n_W):
                horiz_start = w * stride
                horiz_end = w * stride + f

                a_slice_prev = a_prev_pad[vert_start: vert_end, horiz_start: horiz_end, :]

                weights = W[:, :, :]
                z[h, w] = conv_single_step(a_slice_prev, weights)

        if flag == 1:
            z_max.append(np.max(z))
        Z.append(z / z_max[i])

    cache = (A_prev, W, hparameters)

    return Z, z_max, cache

# pooling
def pool_forward(A_prev, hparameters, mode='max'):
    m = len(A_prev)
    f = hparameters['f']
    stride = hparameters['stride']

    A = list()

    for i in range(m):
        (n_H_prev, n_W_prev) = A_prev[i].shape

        n_H = int(1 + (n_H_prev - f) / stride)
        n_W = int(1 + (n_W_prev - f) / stride)

        a = np.zeros((n_H, n_W))

        for h in range(n_H):
            vert_start = h * stride
            vert_end = h * stride + f

            for w in range(n_W):
                horiz_start = w * stride
                horiz_end = w * stride + f

                a_prev_slice = A_prev[i][vert_start: vert_end, horiz_start: horiz_end]

                if mode == 'max':
                    a[h, w] = np.max(a_prev_slice)
                elif mode == 'avg':
                    a[h, w] = np.mean(a_prev_slice)

        A.append(a)

    cache = (A_prev, hparameters)

    return A, cache

# main layer
def borders(images, filter_size=1, pad=0, stride=1, pool_stride=2, pool_size=2, z_max=[]):
    Wv = edges(filter_size, 'vert')
    hparameters = {'pad': pad, 'stride': stride, 'pool_stride': pool_stride, 'f': pool_size, 'z_max': z_max}
    Z, z_max_v, _ = conv_forward(images, Wv, hparameters)

    print('edge filter applied')

    hparameters_pool = {'stride': pool_stride, 'f': pool_size}
    Av, _ = pool_forward(Z, hparameters_pool, mode='max')

    print('vertical filter applied')

    Wh = edges(filter_size, 'horiz')
    hparameters = {'pad': pad, 'stride': stride, 'pool_stride': pool_stride, 'f': pool_size, 'z_max': z_max}
    Z, z_max_h, _ = conv_forward(images, Wh, hparameters)

    print('edge filter applied')

    hparameters_pool = {'stride': pool_stride, 'f': pool_size}
    Ah, _ = pool_forward(Z, hparameters_pool, mode='max')

    print('horizontal filter applied')

    return [(Av[i] + Ah[i]) / 2 for i in range(len(Av))], list(map(np.max, zip(z_max_v, z_max_h)))

# download images from hard disk
def download(directory=PATH):
    start_time = time.time()

    img_names = os.listdir(directory)
    images = list()
    for img in img_names:
        r = plt.imread(directory + img)
        if (len(r[0]) < 1000 and len(r[1]) < 1000 ):
            images.append(r)


    print("--- %s seconds ---" % (time.time() - start_time))
    print(len(images))
    return img_names, images

# calculate borders of original and blurred images
def orig_blur(images, filter_size=1, stride=3, pool_stride=2, pool_size=2, blur=57):
    z_max = []

    img, z_max = borders(images,
                         filter_size=filter_size,
                         stride=stride,
                         pool_stride=pool_stride,
                         pool_size=pool_size
                         )
    print('original image borders is calculated')

    blurred_img = [cv2.GaussianBlur(x, (blur, blur), 0) for x in images]
    print('images blurred')

    blurred, z_max = borders(blurred_img,
                             filter_size=filter_size,
                             stride=stride,
                             pool_stride=pool_stride,
                             pool_size=pool_size,
                             z_max=z_max
                             )
    print('blurred image borders is calculated')

    k = [np.mean(orig) for orig in img]
    print( "orig means:", k)
    return [np.mean(orig) / np.mean(blurred) for (orig, blurred) in zip(img, blurred)]

def clusters_by_clarity():
    """Clusters images by quality"""
    img_names, images = download(PATH)
    k = orig_blur(images, filter_size=1, stride=1)
    print("k = ", k)
    for i in range (len(k)):
        print("k[", i, "]=", k[i])
        if (k[i] < 8):
            try:
                copyfile(PATH + img_names[i], CLARITY_PATH + img_names[i])
            except IOError as io_err:
                os.makedirs(os.path.dirname(CLARITY_PATH+ img_names[i]))
                copyfile(PATH + img_names[i], CLARITY_PATH + img_names[i])
