from pyANNonGPU import PsiDeep, Spins, activation_function, deep_activation, psi_O_k, log_psi_s, ExactSummationSpins #, PauliString

from pytest import approx
import numpy as np
import cmath
import random
import json
from pathlib import Path

translational_invariance = False


def real_noise(shape):
    return 2 * np.random.random_sample(shape) - 1


def complex_noise(shape):
    return real_noise(shape) + 1j * real_noise(shape)


def _test_psi_deep_s(psi_deep, ensemble, gpu):
    psi = psi_deep(gpu)

    use_spins = ensemble.__name__.endswith("Spins")
    if not use_spins and psi.N % 3 != 0:
        return

    N = psi.N if use_spins else psi.N // 3
    psi.num_sites = N

    ensemble = ensemble(N, gpu)

    psi_vector = psi.vector(ensemble)

    local_dim = 2 if use_spins else 4

    def get_input_activation(conf_idx, shift=0):
        if use_spins:
            return np.roll(Spins(conf_idx, 64).array(N), shift)
        else:
            paulis = np.roll(PauliString.enumerate(conf_idx).array(N), shift)

            activations = -np.ones(psi.N)
            for i, p in enumerate(paulis):
                if p == 0:
                    continue

                activations[3 * i + p - 1] = 1

            return activations

    for i in range(10):
        conf_idx = random.randint(0, local_dim**N - 1)

        log_psi_s_ref = 0

        for shift in range(psi.num_sites if psi.symmetric else 1):
            input_activations = get_input_activation(conf_idx, shift)
            activations = +input_activations

            for layer, (w, b) in enumerate(zip(psi.W, psi.b)):
                n, m = len(activations), len(b)

                if m > n:
                    delta = 1
                elif n % m == 0:
                    delta = n // m
                else:
                    delta = w.shape[0]

                fun = activation_function if layer == 0 else deep_activation

                activations = [
                    fun(
                        sum(
                            w[i, j] * activations[(j * delta + i) % n]
                            for i in range(w.shape[0])
                        ) + b[j]
                    )
                    for j in range(len(b))
                ]

            log_psi_s_ref += psi.input_weights @ input_activations + psi.final_weights @ activations

        psi_s_ref = cmath.exp(log_psi_s_ref / (psi.num_sites if psi.symmetric else 1))

        assert psi_vector[conf_idx] == approx(psi_s_ref)


def test_psi_classical_s(psi_classical, gpu):
    psi = psi_classical(gpu)
    num_sites = psi.num_sites

    use_spins = True

    params = 0.1 * complex_noise(psi.num_params)
    # params = np.ones(psi.num_params)
    psi.params = params
    # psi.calibrate(ExactSummationSpins(num_sites, gpu))

    local_dim = 2 if use_spins else 4

    psi_ref = psi.psi_ref

    Basis = Spins  # if use_spins else PauliString

    def local_energy(expr, conf_idx):
        conf_vector = np.zeros(local_dim**num_sites, dtype=complex)
        conf_vector[conf_idx] = 1

        conf = Basis.enumerate(conf_idx)

        log_psi = log_psi_s(psi_ref, conf)

        return sum(
            value.conj() * np.exp(
                log_psi_s(psi_ref, Basis.enumerate(i_prime)) - log_psi
            ) for i_prime, value in enumerate(expr @ conf_vector)
        )

    def get_conf_idx_from_array(a):
        assert use_spins

        return sum((1 if a_i == 1 else 0) * 2**i for i, a_i in enumerate(a))

    H_local = psi.H_local
    if psi.order > 1:
        H_2_local = psi.H_2_local

    # print("params", params)

    for j in range(10):
        conf_idx = random.randint(0, local_dim**num_sites - 1)
        conf = Basis.enumerate(conf_idx)
        # conf_array = conf.array(num_sites)

        log_psi_s_ref = log_psi_s(psi_ref, conf)

        # print("psi_ref:", log_psi_s_ref)

        H_local_energies = np.array([
            sum(
                local_energy(h.roll(shift, num_sites), conf_idx)
                for shift in range(num_sites)
            )
            for h in H_local
        ])

        # print("H_local_energies:", H_local_energies)

        log_psi_s_ref += params[:len(H_local_energies)] @ H_local_energies

        if psi.order > 1:
            H_2_local_energies = np.array([
                sum(
                    local_energy(h.roll(shift, num_sites), conf_idx)
                    for shift in range(num_sites)
                )
                for h in H_2_local
            ])

            # print("len(H_local_energies):", len(H_local_energies))
            # print("len(H_2_local_energies):", len(H_2_local_energies))
            # print("H_2_local_energies:", H_2_local_energies)

            log_psi_s_ref += (
                params[len(H_local_energies): len(H_local_energies) + len(H_2_local_energies)] @
                H_2_local_energies
            )

            H_full_local_energies = np.array([
                [
                    local_energy(h.roll(shift, num_sites), conf_idx)
                    for h in H_local
                ] for shift in range(num_sites)
            ])

            num_pairs = len(H_local_energies) * (len(H_local_energies) + 1) // 2

            k = len(H_local_energies) + len(H_2_local_energies)
            for n in range(
                (psi.num_params - len(H_local_energies) - len(H_2_local_energies)) // num_pairs
            ):
                for l in range(len(H_local_energies)):
                    for l_prime in range(l + 1):
                        log_psi_s_ref += params[k] * sum(
                            H_full_local_energies[delta, l] *
                            H_full_local_energies[(delta + n) % num_sites, l_prime]
                            for delta in range(num_sites)
                        )
                        k += 1

        assert log_psi_s(psi, conf) == approx(log_psi_s_ref)


def _test_O_k(psi_all, gpu):
    psi = psi_all(gpu)

    # use_spins = ensemble.__name__.endswith("Spins")
    # if not use_spins and psi.N % 3 != 0:
    #     return

    # psi.params = 1e-2 * complex_noise(psi.num_params)

    use_spins = True

    eps = 1e-4

    psi_plus = +psi

    def psi_plus_eps(psi, k, eps):
        params = psi.params
        params[k] += eps
        psi_plus.params = params

        return psi_plus

    for i in range(10):
        if use_spins:
            conf_idx = random.randint(0, 2**psi.num_sites - 1)
            conf = Spins.enumerate(conf_idx)
        else:
            conf_idx = random.randint(0, 4**psi.num_sites - 1)
            conf = PauliString.enumerate(conf_idx)

        O_k_vector_ref = np.array([
            (
                log_psi_s(psi_plus_eps(psi, k, eps), conf) -
                log_psi_s(psi_plus_eps(psi, k, -eps), conf)
            ) / (2 * eps)
            for k in range(psi.num_params)
        ])

        if hasattr(psi, "symmetric") and psi.symmetric:
            O_k_vector_test = sum(
                psi_O_k(psi, conf.roll(i, psi.num_sites))
                for i in range(psi.num_sites)
            ) / psi.num_sites
        else:
            O_k_vector_test = psi_O_k(psi, conf)

        passed = np.allclose(O_k_vector_ref, O_k_vector_test, rtol=1e-3, atol=1e-4)

        # print(O_k_vector_test - O_k_vector_ref)

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
