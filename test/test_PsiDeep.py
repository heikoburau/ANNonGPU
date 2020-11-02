from pyANNonGPU import Spins, PauliString, activation_function, psi_O_k, log_psi_s

from pytest import approx
import numpy as np
import cmath
import random
import json
from pathlib import Path

translational_invariance = False


def test_psi_s(psi_deep, ensemble, gpu):
    psi = psi_deep(gpu)

    use_spins = ensemble.__name__.endswith("Spins")
    if not use_spins and psi.N % 3 != 0:
        return

    N = psi.N if use_spins else psi.N // 3
    psi.num_sites = N

    ensemble = ensemble(N, gpu)

    psi_vector = psi.vector(ensemble)

    for i in range(10):
        if use_spins:
            conf_idx = random.randint(0, 2**N - 1)
            spins = Spins(conf_idx, 64).array(N)
            activations = +spins
            input_activations = +spins
        else:
            conf_idx = random.randint(0, 4**N - 1)
            paulis = PauliString.enumerate(conf_idx).array(N)

            activations = -np.ones(psi.N)
            for i, p in enumerate(paulis):
                if p == 0:
                    continue

                activations[3 * i + p - 1] = 1

            input_activations = +activations

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

        print(input_activations)
        print(psi.input_biases)
        log_psi_s_ref = psi.input_biases @ input_activations + psi.final_weights @ activations

        psi_s_ref = cmath.exp(log_psi_s_ref)

        assert psi_vector[conf_idx] == approx(psi_s_ref)


def test_log_psi_s(psi_deep, ensemble, gpu):
    psi = psi_deep(gpu)

    use_spins = ensemble.__name__.endswith("Spins")
    if not use_spins and psi.N % 3 != 0:
        return

    N = psi.N if use_spins else psi.N // 3
    psi.num_sites = N
    ensemble = ensemble(N, gpu)

    eps = 1e-4

    def psi_plus_eps(psi, k, eps):
        psi_plus = +psi
        params = psi_plus.params
        params[k] += eps
        psi_plus.params = params

        return psi_plus

    for i in range(10):
        if use_spins:
            conf_idx = random.randint(0, 2**N - 1)
            conf = Spins.enumerate(conf_idx)
        else:
            conf_idx = random.randint(0, 4**N - 1)
            conf = PauliString.enumerate(conf_idx)

        O_k_vector_ref = np.array([
            (
                log_psi_s(psi_plus_eps(psi, k, eps), conf) -
                log_psi_s(psi_plus_eps(psi, k, -eps), conf)
            ) / (2 * eps)
            for k in range(psi.num_params)
        ])

        O_k_vector_test = psi_O_k(psi, conf)

        passed = np.allclose(O_k_vector_ref, O_k_vector_test, rtol=1e-3, atol=1e-4)

        print(O_k_vector_test - O_k_vector_ref)

        if not passed:
            with open(Path().home() / "test_O_k_vector.json", "w") as f:
                json.dump(
                    {
                        "O_k_vector_test.real": O_k_vector_test.real.tolist(),
                        "O_k_vector_test.imag": O_k_vector_test.imag.tolist(),
                        "O_k_vector_ref.real": O_k_vector_ref.real.tolist(),
                        "O_k_vector_ref.imag": O_k_vector_ref.imag.tolist(),
                    },
                    f
                )

        assert passed
