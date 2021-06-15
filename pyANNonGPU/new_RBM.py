from pyANNonGPU import PsiRBM
import numpy as np
import math


def real_noise(shape):
    return 2 * np.random.random_sample(shape) - 1


def complex_noise(shape):
    return real_noise(shape) + 1j * real_noise(shape)


def new_RBM(
    N,
    M,
    initial_value=(0.01 + 1j * math.pi / 4),
    noise=1e-4,
    gpu=False,
    final_weight=10,
):
    assert M >= N

    W = noise * complex_noise((N, M))

    for alpha in range(M // N):
        W[:, alpha * N: (alpha + 1) * N] += initial_value * np.eye(N)

    return PsiRBM(W, final_weight, 0.0, gpu)
