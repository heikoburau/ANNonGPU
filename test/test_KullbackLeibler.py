from pyANNonGPU import KullbackLeibler, Operator, ExactSummationSpins, new_deep_neural_network
from QuantumExpression import PauliExpression
from pytest import approx
import numpy as np
from pathlib import Path
import json
from math import sqrt


def real_noise(shape):
    return 2 * np.random.random_sample(shape) - 1


def complex_noise(shape):
    return real_noise(shape) + 1j * real_noise(shape)


def test_divergence(psi_classical_ann, gpu):
    psi = psi_classical_ann(gpu)

    esum = ExactSummationSpins(psi.num_sites, gpu)

    L = psi.num_sites
    psi_ann = new_deep_neural_network(L, L, M=[L, L], C=[L, L], noise=1e-2, gpu=gpu)
    psi_ann.calibrate(esum)

    psi.psi_ref = psi_ann
    psi.update_psi_ref_kernel()
    psi.normalize(esum)

    psi.params = 0.1 * complex_noise(len(psi.params))
    psi.normalize(esum)

    kl_divergence = KullbackLeibler(psi.psi_ref.num_params, gpu)
    div_test = kl_divergence(psi, psi.psi_ref, esum)

    psi_vec = psi.vector(esum)
    psi1_vec = psi.psi_ref.vector(esum)
    p = abs(psi_vec)**2

    assert np.sum(p) == approx(1)

    div_ref = (
        p @ abs(np.log(psi1_vec / psi_vec))**2 -
        abs(p @ np.log(psi1_vec / psi_vec))**2
    )

    assert div_test == approx(div_ref)


def __test_gradient(psi_classical_ann, gpu):
    psi = psi_classical_ann(gpu)

    esum = ExactSummationSpins(psi.num_sites, gpu)

    L = psi.num_sites
    psi_ann = new_deep_neural_network(L, L, M=[L, L], C=[L, L], noise=1e-2, gpu=gpu)
    psi_ann.calibrate(esum)

    psi.psi_ref = +psi_ann
    psi.normalize(esum)

    psi.params = 0.1 * complex_noise(len(psi.params))
    psi.normalize(esum)

    kl_divergence = KullbackLeibler(psi.psi_ref.num_params, gpu)

    gradient_test, _ = kl_divergence.gradient(psi, psi_ann, esum, 1)
    params_0 = psi_ann.params

    eps = 1e-6

    def distance_diff(delta_params):
        psi_ann.params = params_0 + delta_params
        plus_distance = kl_divergence(psi, psi_ann, esum)

        psi_ann.params = params_0 - delta_params
        minus_distance = kl_divergence(psi, psi_ann, esum)

        return (plus_distance - minus_distance) / (2 * eps)

    gradient_ref = np.zeros(psi.psi_ref.num_params, dtype=complex)

    for k in range(psi.num_params):
        delta_params = np.zeros_like(params_0)
        delta_params[k] = eps
        gradient_ref[k] = distance_diff(delta_params)

        delta_params = np.zeros_like(params_0)
        delta_params[k] = 1j * eps
        gradient_ref[k] += 1j * distance_diff(delta_params)

    print(gradient_ref - gradient_test)
    print(gradient_test)

    passed = np.allclose(gradient_ref, gradient_test, rtol=1e-3, atol=1e-4)

    if not passed:
        with open(Path().home() / "test_gradient.json", "w") as f:
            json.dump(
                {
                    "gradient_test.real": gradient_test.real.tolist(),
                    "gradient_test.imag": gradient_test.imag.tolist(),
                    "gradient_ref.real": gradient_ref.real.tolist(),
                    "gradient_ref.imag": gradient_ref.imag.tolist(),
                },
                f
            )

    assert passed


def __test_gradient2(psi_deep, ensemble, hamiltonian, gpu):
    psi_0 = psi_deep(gpu)
    if psi_0.symmetric:
        return

    psi_0.params = (2 * np.random.rand(psi_0.num_params)) * psi_0.params
    psi = psi_deep(gpu)

    num_sites = psi.num_sites

    use_spins = ensemble.__name__.endswith("Spins")
    if not use_spins and psi.N != 3 * num_sites:
        return

    ensemble = ensemble(num_sites, gpu)

    psi_0.normalize(ensemble)
    psi.normalize(ensemble)

    psi_1 = +psi

    # print(psi.num_params)

    hs_distance = HilbertSpaceDistance(psi.num_params, gpu)
    op = Operator(PauliExpression(1), gpu)

    gradient_test, distance = hs_distance.gradient(psi_0, psi_1, op, True, ensemble, 1)

    eps = 1e-6

    def distance_diff(delta_params):
        psi_1.params = psi.params + delta_params
        plus_distance = hs_distance(psi_0, psi_1, op, True, ensemble)

        psi_1.params = psi.params - delta_params
        minus_distance = hs_distance(psi_0, psi_1, op, True, ensemble)

        return (plus_distance - minus_distance) / (2 * eps)

    gradient_ref = np.zeros(psi.num_params, dtype=complex)

    for k in range(psi.num_params):
        delta_params = np.zeros(psi.num_params, dtype=complex)
        delta_params[k] = eps
        gradient_ref[k] = distance_diff(delta_params)

        delta_params = np.zeros(psi.num_params, dtype=complex)
        delta_params[k] = 1j * eps
        gradient_ref[k] += 1j * distance_diff(delta_params)

    print("distance:", distance)
    print(gradient_test - gradient_ref)
    print(gradient_test)
    print(gradient_ref)

    passed = np.allclose(gradient_ref, gradient_test, rtol=1e-3, atol=1e-4)

    if not passed:
        with open(Path().home() / "test_gradient.json", "w") as f:
            json.dump(
                {
                    "gradient_test.real": gradient_test.real.tolist(),
                    "gradient_test.imag": gradient_test.imag.tolist(),
                    "gradient_ref.real": gradient_ref.real.tolist(),
                    "gradient_ref.imag": gradient_ref.imag.tolist(),
                },
                f
            )

    assert passed


# def test_gradient2(psi_all, hamiltonian, gpu):
#     psi = psi_all(gpu)

#     N = psi.N
#     H = hamiltonian(N)
#     spin_ensemble = ExactSummationSpins(N, gpu)

#     psi.normalize(spin_ensemble)
#     psi1 = +psi

#     hs_distance = HilbertSpaceDistance(psi.num_params, gpu)
#     op = Operator(1j * psi.transform(H) * 1e-2, gpu)

#     gradient_test, _ = hs_distance.gradient(psi, psi, op, False, spin_ensemble)
#     gradient_test[:2 * N] = gradient_test[:2 * N].real

#     eps = 1e-6

#     def distance_diff(delta_params):
#         psi1.params = psi.params + delta_params
#         plus_distance = hs_distance(psi, psi1, op, False, spin_ensemble)

#         psi1.params = psi.params - delta_params
#         minus_distance = hs_distance(psi, psi1, op, False, spin_ensemble)

#         return (plus_distance - minus_distance) / (2 * eps)

#     gradient_ref = np.zeros(psi.num_params, dtype=complex)

#     for k in range(psi.num_params):
#         delta_params = np.zeros(psi.num_params, dtype=complex)
#         delta_params[k] = eps
#         gradient_ref[k] = distance_diff(delta_params)

#         if k >= 2 * N:
#             delta_params = np.zeros(psi.num_params, dtype=complex)
#             delta_params[k] = 1j * eps
#             gradient_ref[k] += 1j * distance_diff(delta_params)

#     print(gradient_ref - gradient_test)
#     print(gradient_test)

#     passed = np.allclose(gradient_ref, gradient_test, rtol=1e-3, atol=1e-8)

#     if not passed:
#         with open(Path().home() / "test_gradient.json", "w") as f:
#             json.dump(
#                 {
#                     "gradient_test.real": gradient_test.real.tolist(),
#                     "gradient_test.imag": gradient_test.imag.tolist(),
#                     "gradient_ref.real": gradient_ref.real.tolist(),
#                     "gradient_ref.imag": gradient_ref.imag.tolist(),
#                 },
#                 f
#             )

#     assert passed
