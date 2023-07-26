from typing import Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as func

from tools.cross_correlation import xcorr_torch
from tools.spectral_tools import gen_mtf

from math import floor


def local_corr_mask(img_in, ratio, sensor, device, kernel=8):
    """
        Compute the threshold mask for the structural loss.

        Parameters
        ----------
        img_in : Torch Tensor
            The test image, already normalized and with the MS part upsampled with ideal interpolator.
        ratio : int
            The resolution scale which elapses between MS and PAN.
        sensor : str
            The name of the satellites which has provided the images.
        device : Torch device
            The device on which perform the operation.
        kernel : int
            The semi-width for local cross-correlation computation.
            (See the cross-correlation function for more details)

        Return
        ------
        mask : PyTorch Tensor
            Local correlation field stack, composed by each MS and PAN. Dimensions: Batch, B, H, W.

        """

    pan = torch.unsqueeze(img_in[:, -1, :, :], dim=1)
    ms = img_in[:, :-1, :, :]

    mtf_kern = gen_mtf(ratio, sensor)[:, :, 0]
    mtf_kern = np.expand_dims(mtf_kern, axis=(0, 1))
    mtf_kern = torch.from_numpy(mtf_kern).type(torch.float32)
    pad = floor((mtf_kern.shape[-1] - 1) / 2)

    padding = nn.ReflectionPad2d(pad)

    depthconv = nn.Conv2d(in_channels=1,
                          out_channels=1,
                          groups=1,
                          kernel_size=mtf_kern.shape,
                          bias=False)

    depthconv.weight.data = mtf_kern
    depthconv.weight.requires_grad = False
    depthconv.to(device)
    pan = pan.to(device)
    ms = ms.to(device)
    pan = padding(pan)
    pan = depthconv(pan)
    mask = xcorr_torch(pan, ms, kernel)
    mask = 1.0 - mask

    return mask.float().to(device)


def parameters_def(sensor, lr, epochs):

    if sensor == 'WV3':
        # Defining Sensor characteristics
        nbands = 8
        nbits = 11

        # HyperParameters
        learning_rate = 0.00004

        alpha = 0.05
        beta = 1.25
        gamma = 5.00
    elif sensor == 'WV2':
        # Defining Sensor characteristics
        nbands = 8
        nbits = 11

        # HyperParameters
        learning_rate = 0.000035
        alpha = 0.05
        beta = 1.0
        gamma = 6.00
    elif sensor == 'GE1':
        # Defining Sensor characteristics
        nbands = 4
        nbits = 11

        # HyperParameters
        learning_rate = 0.00005
        alpha = 0.075
        beta = 1.5
        gamma = 7.0

    else:
        raise ValueError('Sensor not supported')

    if lr != -1.0:
        learning_rate = lr

    if epochs == -1:
        epochs = 100

    return nbands, nbits, learning_rate, alpha, beta, gamma, epochs


class ImageClassification(nn.Module):
    def __init__(
        self,
        *,
        crop_size: int,
        resize_size: int = 256,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        self.crop_size = [crop_size]
        self.resize_size = [resize_size]
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation

    def forward(self, img: Tensor) -> Tensor:
        img = func.resize(img, self.resize_size, interpolation=self.interpolation)
        img = func.center_crop(img, self.crop_size)
        if not isinstance(img, Tensor):
            img = func.pil_to_tensor(img)
        img = func.convert_image_dtype(img, torch.float)
        img = func.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    crop_size={self.crop_size}"
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            f"The images are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``, "
            f"followed by a central crop of ``crop_size={self.crop_size}``. Finally the values are first rescaled to "
            f"``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and ``std={self.std}``."
        )
