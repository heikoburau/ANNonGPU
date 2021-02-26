from pyANNonGPU import PsiCNN


import numpy as np
import math


def real_noise(shape):
    return 2 * np.random.random_sample(shape) - 1


def complex_noise(shape):
    return real_noise(shape) + 1j * real_noise(shape)


def noise_vector(shape, real=True):
    if real:
        return real_noise(shape)
    else:
        return complex_noise(shape)


def new_convolutional_network(
    num_sites,
    N,
    layers,
    initial_value=(0.01 + 1j * math.pi / 4),
    noise=1e-4,
    final_factor=10,
    real=False,
    gpu=False
):
    """
    A proper choice of 'final_factor' seems to be very important. For instance, for tdvp with 12 spins a value of 20 works well, but anything below fails.
    """

    assert num_sites == N or 3 * num_sites == N

    num_channels_list = np.array(
        list(zip(*layers))[0]
    )
    connectivity_list = np.array(
        list(zip(*layers))[1]
    )

    params = []
    for layer, (num_channels, connectivity) in enumerate(layers):
        assert connectivity <= N

        num_channel_links = num_channels * (layers[layer - 1][0] if layer > 0 else 1)

        channel_link = noise * noise_vector(connectivity, real)
        if layer == 0:
            channel_link[connectivity // 2] = initial_value.real if real else initial_value

        params += num_channel_links * list(channel_link)
    params = np.array(params)

    return PsiCNN(num_sites, N, num_channels_list, connectivity_list, params, final_factor, 1, gpu)
