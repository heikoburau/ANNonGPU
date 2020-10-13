from pyANNonGPU import Spins, activation_function
from pytest import approx
import numpy as np
import cmath
import random

translational_invariance = False


def test_psi_s(psi_deep, gpu):
    psi = psi_deep(gpu)
    psi_vector = psi.vector

    N = psi.N

    for i in range(10):
        spins_idx = random.randint(0, 2**N - 1)
        spins = Spins(spins_idx, 64).array(N)

        activations = +spins
        for w, b in zip(psi.W, psi.b):
            n, m = len(activations), len(b)

            if m > n:
                delta = 1
            elif n % m == 0:
                delta = n // m
            else:
                delta = w.shape[0]

            activations = [
                activation_function(
                    sum(
                        w[i, j] * activations[(j * delta + i) % n]
                        for i in range(w.shape[0])
                    ) + b[j]
                )
                for j in range(len(b))
            ]

        print(spins)
        print(psi.input_biases)
        log_psi_s_ref = psi.input_biases @ spins + psi.final_weights @ activations

        psi_s_ref = cmath.exp(log_psi_s_ref)

        assert psi_vector[spins_idx] == approx(psi_s_ref)
