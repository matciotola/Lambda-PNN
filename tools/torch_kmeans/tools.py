import numpy as np
import math


def regular_chessboard_initialization(samples, num_clusters):
    """
    initialize cluster centers
    :param samples: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(samples)
    x = int(np.floor(np.sqrt(num_samples)))

    rows = int(np.floor(np.sqrt(x)))

    ini_offset = x // 8 - 1
    f = x - ini_offset - 1
    if f % 2 != 0:
        f = f - 1
    ini = np.floor(np.linspace(ini_offset, f, rows)).astype(int)
    loop = math.ceil(num_clusters / len(ini))
    offset = x * (x // loop)
    ini_offset = x*(((x-1) % (x // loop)) // 2)

    ini = ini + ini_offset
    indices = [ini]

    for i in range(1, loop):
        indices.append(ini + i*offset)

    indices = np.asarray(indices).flatten()
    initial_state = samples[indices]
    return initial_state
