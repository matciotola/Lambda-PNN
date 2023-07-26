import math
import numpy as np
import scipy.ndimage.filters as ft
import torch
import torch.nn as nn


cdf23 = np.asarray(
        [0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0,
         -0.000060081482])
cdf23 = [element * 2 for element in cdf23]


def interp23tap(img, ratio):
    """
        Polynomial (with 23 coefficients) interpolator Function.

        For more details please refers to:

        [1]  B. Aiazzi, L. Alparone, S. Baronti, and A. Garzelli - Context-driven fusion of high spatial and spectral
             resolution images based on oversampled multiresolution analysis
        [2] B. Aiazzi, S. Baronti, M. Selva, and L. Alparone - Bi-cubic interpolation for shift-free pan-sharpening
        [3] G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. Orn Ulfarsson, L. Alparone, J. Chanussot -
            A new benchmark based on recent advances in multispectral pansharpening: Revisiting pansharpening with
            classical and emerging pansharpening methods


        Parameters
        ----------
        img : Numpy Array
            Image to be scaled. Dimension: H, W, B
        ratio : int
            The desired scale. It must be a factor power of 2.


        Return
        ------
        img : Numpy array
            the interpolated img.

        """

    assert ((2 ** (round(math.log(ratio, 2)))) == ratio), 'Error: Only resize factors power of 2'
    r, c, b = img.shape

    base_coeff = np.expand_dims(np.concatenate([np.flip(cdf23[1:]), cdf23]), axis=-1)

    for z in range(int(ratio / 2)):

        i1_lru = np.zeros(((2 ** (z + 1)) * r, (2 ** (z + 1)) * c, b))

        if z == 0:
            i1_lru[1::2, 1::2, :] = img
        else:
            i1_lru[::2, ::2, :] = img

        for i in range(b):
            temp = ft.convolve(np.transpose(i1_lru[:, :, i]), base_coeff, mode='wrap')
            i1_lru[:, :, i] = ft.convolve(np.transpose(temp), base_coeff, mode='wrap')

        img = i1_lru

    return img


def interp23tap_torch(img, ratio, device):
    """
        A PyTorch implementation of the Polynomial interpolator Function.

        For more details please refers to:

        [1]  B. Aiazzi, L. Alparone, S. Baronti, and A. Garzelli - Context-driven fusion of high spatial and spectral
             resolution images based on oversampled multiresolution analysis
        [2] B. Aiazzi, S. Baronti, M. Selva, and L. Alparone - Bi-cubic interpolation for shift-free pan-sharpening
        [3] G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. Orn Ulfarsson, L. Alparone, J. Chanussot -
            A new benchmark based on recent advances in multispectral pansharpening: Revisiting pansharpening with
            classical and emerging pansharpening methods


        Parameters
        ----------
        img : Numpy Array
            Image to be scaled. The conversion in Torch Tensor is made within the function. Dimension: H, W, B
        ratio : int
            The desired scale. It must be a factor power of 2.
        device : Torch Device
            The device on which perform the operation.


        Return
        ------
        img : Numpy array
           The interpolated img.

    """

    assert ((2 ** (round(math.log(ratio, 2)))) == ratio), 'Error: Only resize factors power of 2'

    bs, b, r, c = img.shape

    base_coeff = np.expand_dims(np.concatenate([np.flip(cdf23[1:]), cdf23]), axis=-1)
    base_coeff = np.expand_dims(base_coeff, axis=(0, 1))
    base_coeff = np.concatenate([base_coeff] * b, axis=0)

    base_coeff = torch.from_numpy(base_coeff).to(device)

    for z in range(int(ratio / 2)):

        i1_lru = torch.zeros((bs, b, (2 ** (z + 1)) * r, (2 ** (z + 1)) * c), device=device, dtype=base_coeff.dtype)

        if z == 0:
            i1_lru[:, :, 1::2, 1::2] = img
        else:
            i1_lru[:, :, ::2, ::2] = img

        conv = nn.Conv2d(in_channels=b, out_channels=b, padding=(11, 0),
                         kernel_size=base_coeff.shape, groups=b, bias=False, padding_mode='circular')

        conv.weight.data = base_coeff
        conv.weight.requires_grad = False

        t = conv(torch.transpose(i1_lru, 2, 3))
        img = conv(torch.transpose(t, 2, 3))

    return img
