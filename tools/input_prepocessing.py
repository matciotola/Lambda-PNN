import numpy as np
from skimage import transform
import math
import torch
from torch import nn

from tools.interpolator_tools import interp23tap_torch
import tools.spectral_tools as ut


def input_preparation(ms, pan, ratio, nbits, device):
    """
        Prepare the remote sensing imagery for the pansharpening algorithm.
        In particular, the MS is upsampled with an ideal filter to the scale of PAN and a unique stack is created from
        both. After this, normalization is performed.

        Parameters
        ----------
        ms : Numpy Array
            stack of Multi-Spectral bands. Dimension: H, W, B
        pan : Numpy Array
            Panchromatic Band converted in Numpy Array. Dimensions: H, W
        ratio : int
            the resolution scale which elapses between MS and PAN.
        nbits : int
            the number of bits with which the images have been codified.
        device : PyTorch Device object
            the device on which the algorithm will be performed


        Return
        ------
        img_in : Tensor
            the stack of MS + PAN images normalized as I = (I / 2 ** nbits)

        ms : Tensor
            the original MS converted in PyTorch Tensor and moved to the device.

        ms_exp : Tensor
            the MS upsampled with an ideal filter to the scale of PAN converted in PyTorch Tensor
            and moved to the device.

        pan : Tensor
            the PAN image converted in PyTorch Tensor and moved to the device.

        """

    ms = ms.astype(np.float32)
    pan = pan.astype(np.float32)
    max_value = 2 ** nbits
    ms = torch.tensor(np.moveaxis(ms, -1, 0)[None, :, :, :]).float().to(device)
    ms_exp = interp23tap_torch(ms.double(), ratio, device).float()

    pan = torch.tensor(pan[None, None]).to(device)

    img_in = torch.cat((ms_exp, pan), dim=1) / max_value

    ms = ms / max_value
    ms_exp = ms_exp / max_value
    pan = pan / max_value

    return img_in, ms, ms_exp, pan


def resize_images(img_ms, img_pan, ratio, sensor=None, mtf=None, apply_mtf_to_pan=False):
    """
        Function to perform downscale of all the data provided by the satellite.
        It downsamples the data of the scale value.
        To more detail please refer to

        [1] G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. Orn Ulfarsson, L. Alparone, J. Chanussot -
            A new benchmark based on recent advances in multispectral pansharpening: Revisiting pansharpening with
            classical and emerging pansharpening methods
        [2] L. Wald, (1) T. Ranchin, (2) M. Mangolini - Fusion of satellites of different spatial resolutions:
            assessing the quality of resulting images
        [3] B. Aiazzi, L. Alparone, S. Baronti, A. Garzelli, M. Selva - MTF-tailored Multiscale Fusion of
            High-resolution MS and Pan Imagery
        [4] M. Ciotola, S. Vitale, A. Mazza, G. Poggi, G. Scarpa - Pansharpening by convolutional neural networks in
            the full resolution framework


        Parameters
        ----------
        img_ms : Numpy Array
            stack of Multi-Spectral bands. Dimension: H, W, B
        img_pan : Numpy Array
            Panchromatic Band converted in Numpy Array. Dimensions: H, W
        ratio : int
            the resolution scale which elapses between MS and PAN.
        sensor : str
            The name of the satellites which has provided the images.
        mtf : Dictionary
            The desired Modulation Transfer Frequencies with which perform the low pass filtering process.
            Example of usage:
                MTF = {'nyquist_gains' : np.asarray([0.21, 0.2, 0.3, 0.4]), 'pan_nyquist_gains': 0.5}
        apply_mtf_to_pan : bool
            Activate the downsample of the Panchromatic band with the Modulation Transfer Function protocol
            (Actually this feature is not used in our algorithm).


        Return
        ------
        ms_lr : Numpy array
            the stack of Multi-Spectral bands downgraded by the ratio factor
        pan_lr : Numpy array
            The panchromatic band downsampled by the ratio factor

        """
    nyquist_gains = []
    pan_nyquist_gains = []
    if (sensor is None) & (mtf is None):
        ms_scale = (img_ms.shape[0] // ratio, img_ms.shape[1] // ratio, img_ms.shape[2])
        pan_scale = (img_pan.shape[0] // ratio, img_pan.shape[1] // ratio)
        ms_lr = transform.resize(img_ms, ms_scale, order=3)
        pan_lr = transform.resize(img_pan, pan_scale, order=3)

        return ms_lr, pan_lr

    elif (sensor == 'QB') & (mtf is None):
        nyquist_gains = np.asarray([0.34, 0.32, 0.30, 0.22])  # Bands Order: B,G,R,NIR
        pan_nyquist_gains = np.asarray([0.15])
    elif ((sensor == 'Ikonos') or (sensor == 'IKONOS')) & (mtf is None):
        nyquist_gains = np.asarray([0.26, 0.28, 0.29, 0.28])  # Bands Order: B,G,R,NIR
        pan_nyquist_gains = np.asarray([0.17])
    elif (sensor == 'GeoEye1' or sensor == 'GE1') & (mtf is None):
        nyquist_gains = np.asarray([0.23, 0.23, 0.23, 0.23])  # Bands Order: B, G, R, NIR
        pan_nyquist_gains = np.asarray([0.16])
    elif (sensor == 'WV2') & (mtf is None):
        nyquist_gains = 0.35 * np.ones((1, 7))
        nyquist_gains = np.append(nyquist_gains, 0.27)
        pan_nyquist_gains = np.asarray([0.11])
    elif (sensor == 'WV3') & (mtf is None):
        nyquist_gains = [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315]
        pan_nyquist_gains = np.asarray([0.5])
    elif mtf is not None:
        nyquist_gains = mtf['nyquist_gains']
        pan_nyquist_gains = np.asarray([mtf['pan_nyquist_gains']])

    n = 41

    b = img_ms.shape[-1]

    img_ms = np.moveaxis(img_ms, -1, 0)
    img_ms = np.expand_dims(img_ms, axis=0)

    h = ut.nyquist_filter_generator(nyquist_gains, ratio, n)
    h = ut.mtf_kernel_to_torch(h)

    conv = nn.Conv2d(in_channels=b, out_channels=b, padding=math.ceil(n / 2),
                     kernel_size=h.shape, groups=b, bias=False, padding_mode='replicate')

    conv.weight.data = h
    conv.weight.requires_grad = False

    ms_lp = conv(torch.from_numpy(img_ms)).numpy()
    ms_lp = np.squeeze(ms_lp)
    ms_lp = np.moveaxis(ms_lp, 0, -1)
    ms_scale = (math.floor(ms_lp.shape[0] / ratio), math.floor(ms_lp.shape[1] / ratio), ms_lp.shape[2])
    pan_scale = (math.floor(img_pan.shape[0] / ratio), math.floor(img_pan.shape[1] / ratio))

    ms_lr = transform.resize(ms_lp, ms_scale, order=0)

    if apply_mtf_to_pan:
        img_pan = np.expand_dims(img_pan, [0, 1])

        h = ut.nyquist_filter_generator(pan_nyquist_gains, ratio, n)
        h = ut.mtf_kernel_to_torch(h)

        conv = nn.Conv2d(in_channels=1, out_channels=1, padding=math.ceil(n / 2),
                         kernel_size=h.shape, groups=1, bias=False, padding_mode='replicate')

        conv.weight.data = h
        conv.weight.requires_grad = False

        pan_lp = conv(torch.from_numpy(img_pan)).numpy()
        pan_lp = np.squeeze(pan_lp)
        pan_lr = transform.resize(pan_lp, pan_scale, order=0)

    else:
        pan_lr = transform.resize(img_pan, pan_scale, order=3)

    return ms_lr, pan_lr
