from pyANNonGPU import PsiDeep, PsiDeepSigned


import numpy as np
import math
from itertools import product


def real_noise(shape):
    return 2 * np.random.random_sample(shape) - 1


def complex_noise(shape):
    return real_noise(shape) + 1j * real_noise(shape)


def noise_vector(shape, real=True):
    if real:
        return real_noise(shape)
    else:
        return complex_noise(shape)


def new_deep_neural_network(
    num_sites,
    N,
    M,
    C,
    initial_value=(0.01 + 1j * math.pi / 4),
    a=0,
    noise=1e-4,
    gpu=False,
    noise_modulation="auto",
    final_weights=10,
    signed=False
):
    if signed:
        return PsiDeepSigned(
            new_deep_neural_network(num_sites, N, M, C, initial_value.real, a, noise, gpu, noise_modulation, final_weights, False),
            new_deep_neural_network(num_sites, N, M, C, initial_value.real, a, noise, gpu, noise_modulation, final_weights, False)
        )

    dim = len(N) if isinstance(N, (list, tuple)) else 1

    N_linear = N if dim == 1 else N[0] * N[1]
    M_linear = M if dim == 1 else [m[0] * m[1] for m in M]
    C_linear = C if dim == 1 else [c[0] * c[1] for c in C]

    for n, m, c in zip([N] + M[:-1], M, C):
        if dim == 1:
            assert (m * c) % n == 0
            assert c <= n
        elif dim == 2:
            assert (m[0] * c[0]) % n[0] == 0
            assert (m[1] * c[1]) % n[1] == 0
            assert c[0] <= n[0]
            assert c[1] <= n[1]

    if isinstance(a, (float, int, complex)):
        a = a * np.ones(N_linear)
    else:
        a = np.array(a)

    is_real = (initial_value.imag == 0)

    b = [noise * noise_vector(m, is_real) for m in M_linear]
    w = noise * noise_vector((C_linear[0], M_linear[0]), is_real)
    w[C_linear[0] // 2, :] += initial_value
    W = [w]

    if noise_modulation == "auto":
        noise_modulation = []
        for c, m, next_c in zip(C_linear[1:], M_linear[1:], C_linear[2:] + [1]):
            noise_modulation.append(math.sqrt(6 / (c + next_c)))

    for k, (c, m, next_c, nm) in enumerate(zip(C_linear[1:], M_linear[1:], C_linear[2:] + [1], noise_modulation)):
        w = (
            # math.sqrt(6 / (c + next_c)) * real_noise((c, m)) +
            # Wenn hier complex noise verwendet wird, wirds unbrauchbar. Es scheint die Lokalitaet zu zerstoeren.
            math.sqrt(6 / (c + next_c)) * real_noise((c, m)) +
            # 1j * math.sqrt(6 / (c + next_c)) / 1e2 * real_noise((c, m)) +
            noise * noise_vector((c, m), is_real)
        )
        W.append(w)

    def delta_func(n, m, c):
        if m > n:
            return 1
        elif n % m == 0:
            return n // m
        else:
            return c

    connections = []
    for n, m, c in zip([N] + M[:-1], M, C):
        if dim == 1:
            delta_j = delta_func(n, m, c)

            connections.append(np.array([
                [
                    (j * delta_j + i) % n
                    for j in range(m)
                ]
                for i in range(c)
            ]))
        elif dim == 2:
            range2D = lambda area: product(range(area[0]), range(area[1]))
            linear_idx = lambda row, col: row * n[0] + col

            delta_j1 = delta_func(n[0], m[0], c[0])
            delta_j2 = delta_func(n[1], m[1], c[1])

            connections.append(np.array([
                [
                    linear_idx(
                        (j1 * delta_j1 + i1) % n[0],
                        (j2 * delta_j2 + i2) % n[1]
                    )
                    for j1, j2 in range2D(m)
                ]
                for i1, i2 in range2D(c)
            ]))

    if isinstance(final_weights, (float, int)):
        final_weights = final_weights * np.ones(M_linear[-1])
    assert len(final_weights) == M_linear[-1]

    assert (num_sites == len(a)) or (3 * num_sites == len(a)), (
        f"mismatch between num_sites = {num_sites} and len(a) = {len(a)}."
    )

    return PsiDeep(num_sites, a, b, connections, W, final_weights, 1.0, gpu)
