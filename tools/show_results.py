from matplotlib import pyplot as plt
import numpy as np
from skimage.transform import resize
from tools.interpolator_tools import interp23tap


def show(starting_img_ms, img_pan, algorithm_outcome, ratio, method, q_min=0.02, q_max=0.98):

    q_ms = np.quantile(starting_img_ms, (q_min, q_max), (0, 1), keepdims=True)
    q_pan = np.quantile(img_pan, (q_min, q_max), (0, 1), keepdims=True)

    ms_shape = (starting_img_ms.shape[0] * ratio, starting_img_ms.shape[1] * ratio, starting_img_ms.shape[2])

    ms_lr_4x = resize(starting_img_ms, ms_shape, order=0)
    ms_exp = interp23tap(starting_img_ms, ratio)

    dp = algorithm_outcome - ms_exp
    q_d = np.quantile(abs(dp), q_max, (0, 1))
    if starting_img_ms.shape[-1] == 8:
        rgb = (4, 2, 1)
        ryb = (4, 3, 1)
    else:
        rgb = (2, 1, 0)
        ryb = (2, 3, 0)
    plt.figure()
    ax1 = plt.subplot(2, 4, 1)
    pan_t = (img_pan - q_pan[0, :, :]) / (q_pan[1, :, :] - q_pan[0, :, :])
    pan_t = np.clip(pan_t, 0, 1)
    plt.imshow(pan_t, cmap='gray')
    ax1.set_title('PAN')

    transf = (ms_lr_4x - q_ms[0, :, :]) / (q_ms[1, :, :] - q_ms[0, :, :])
    transf = np.clip(transf, 0, 1)

    ax2 = plt.subplot(2, 4, 2, sharex=ax1, sharey=ax1)
    plt.imshow(transf[:, :, rgb])
    ax2.set_title('MS (RGB)')

    ax6 = plt.subplot(2, 4, 6, sharex=ax1, sharey=ax1)
    plt.imshow(transf[:, :, ryb])
    ax6.set_title('MS (RYB)')

    transf = (algorithm_outcome - q_ms[0, :, :]) / (q_ms[1, :, :] - q_ms[0, :, :])
    transf = np.clip(transf, 0, 1)

    ax3 = plt.subplot(2, 4, 3, sharex=ax1, sharey=ax1)
    plt.imshow(transf[:, :, rgb])
    ax3.set_title(method + ' (RGB)')

    ax7 = plt.subplot(2, 4, 7, sharex=ax1, sharey=ax1)
    plt.imshow(transf[:, :, ryb])
    ax7.set_title(method + ' (RYB)')

    transf = 0.5 + dp / (2 * q_d)
    transf = np.clip(transf, 0, 1)

    ax4 = plt.subplot(2, 4, 4, sharex=ax1, sharey=ax1)
    plt.imshow(transf[:, :, rgb])
    ax4.set_title('Detail (RGB)')

    ax8 = plt.subplot(2, 4, 8, sharex=ax1, sharey=ax1)
    plt.imshow(transf[:, :, ryb])
    ax8.set_title('Detail (RYB)')
    plt.show()
    return
