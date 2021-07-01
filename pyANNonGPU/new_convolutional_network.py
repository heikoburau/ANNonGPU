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


def prod(x_list):
    r = 1
    for x in x_list:
        r *= x
    return r


def new_convolutional_network(
    L,
    layers,
    initial_value=(0.01 + 1j * math.pi / 4),
    noise=1e-4,
    final_factor=10,
    symmetry_classes=None,
    real=False,
    gpu=False
):
    """
    A proper choice of 'final_factor' seems to be very important. For instance, for tdvp with 12 spins a value of 20 works well, but anything below fails.
    """

    num_channels_list = np.array(
        list(zip(*layers))[0]
    )
    connectivity_list = np.array(
        list(zip(*layers))[1]
    )
    if symmetry_classes is None:
        symmetry_classes = np.zeros(prod(L))
    num_symmetry_classes = len(set(symmetry_classes))

    params = []
    for layer, (num_channels, ndim_connectivity) in enumerate(layers):
        for c, l in zip(ndim_connectivity, L):
            assert c <= l

        connectivity = prod(ndim_connectivity)

        num_prev_channels = layers[layer - 1][0] if layer > 0 else 1
        num_next_channels = layers[layer + 1][0] if layer < len(layers) - 1 else 1
        next_connectivity = prod(layers[layer + 1][1]) if layer < len(layers) - 1 else 1

        num_channel_links = num_channels * num_prev_channels

        for cl in range(num_channel_links):
            channel_link = noise * noise_vector(connectivity, real)
            if layer == 0:
                channel_link[connectivity // 2] = initial_value.real if real else initial_value
            else:
                channel_link += math.sqrt(
                    6 / (connectivity * num_prev_channels + next_connectivity * num_next_channels)
                ) * real_noise(connectivity)

            params += num_symmetry_classes * list(channel_link)

    params = np.array(params)

    return PsiCNN(L, num_channels_list, connectivity_list, symmetry_classes, params, final_factor, 0, gpu)
