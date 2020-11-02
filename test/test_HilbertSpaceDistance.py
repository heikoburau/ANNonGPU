from pyANNonGPU import HilbertSpaceDistance, Operator
from QuantumExpression import PauliExpression
from pytest import approx
import numpy as np
from pathlib import Path
import json
from math import sqrt


def test_distance1(psi_deep, ensemble, hamiltonian, gpu):
    psi = psi_deep(gpu)

    use_spins = ensemble.__name__.endswith("Spins")
    if not use_spins and psi.N % 3 != 0:
        return

    N = psi.N if use_spins else psi.N // 3

    ensemble = ensemble(N, gpu)

    H = hamiltonian(N)

    psi.normalize(ensemble)

    t = 1e-2
    hs_distance = HilbertSpaceDistance(psi.num_params, gpu)
    op = Operator(1 + 1j * H * t, gpu)
    distance_test = hs_distance(psi, psi, op, True, ensemble)

    psi_vector = psi.vector(ensemble)

    psi_prime_vector = (1 + 1j * H * t).matrix(N, "spins" if use_spins else "paulis") @ psi_vector
    psi_prime_vector /= np.linalg.norm(psi_prime_vector)

    distance_ref = sqrt(1.0 - abs(np.vdot(
        psi_vector,
        psi_prime_vector
    ))**2)

    assert distance_test == approx(distance_ref, rel=1e-3, abs=1e-7)


# def test_distance2(psi_all, hamiltonian, gpu):
#     psi = psi_all(gpu)

#     N = psi.N
#     H = hamiltonian(N)
#     spin_ensemble = ExactSummationSpins(N, gpu)

#     psi.normalize(spin_ensemble)
#     psi1 = +psi
#     psi1.params = 0.95 * psi1.params
#     psi1.normalize(spin_ensemble)

#     t = 1e-4
#     hs_distance = HilbertSpaceDistance(psi.num_params, gpu)
#     op = Operator(1j * psi.transform(H) * t, gpu)
#     distance_test = hs_distance(psi, psi1, op, False, spin_ensemble)

#     H_diag = np.linalg.eigh(H.matrix(N))
#     U_t = H_diag[1] @ (np.exp(1j * H_diag[0] * t) * H_diag[1]).T

#     distance_ref = sqrt(1.0 - abs(np.vdot(psi.vector(spin_ensemble), U_t @ psi1.vector))**2)

#     assert distance_test == approx(distance_ref, rel=1e-1, abs=1e-8)


def test_gradient1(psi_deep, ensemble, single_sigma, gpu):
    psi = psi_deep(gpu)

    use_spins = ensemble.__name__.endswith("Spins")
    if not use_spins and psi.N % 3 != 0:
        return

    N = psi.N if use_spins else psi.N // 3

    ensemble = ensemble(N, gpu)

    expr = single_sigma(N)

    psi.normalize(ensemble)
    psi1 = +psi

    hs_distance = HilbertSpaceDistance(psi.num_params, gpu)
    op = Operator(expr, gpu)

    gradient_test, _ = hs_distance.gradient(psi, psi, op, True, ensemble, 1)

    eps = 1e-6

    def distance_diff(delta_params):
        psi1.params = psi.params + delta_params
        plus_distance = hs_distance(psi, psi1, op, True, ensemble)

        psi1.params = psi.params - delta_params
        minus_distance = hs_distance(psi, psi1, op, True, ensemble)

        return (plus_distance - minus_distance) / (2 * eps)

    gradient_ref = np.zeros(psi.num_params, dtype=complex)

    for k in range(psi.num_params):
        delta_params = np.zeros(psi.num_params, dtype=complex)
        delta_params[k] = eps
        gradient_ref[k] = distance_diff(delta_params)

        delta_params = np.zeros(psi.num_params, dtype=complex)
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


def test_gradient2(psi_deep, ensemble, hamiltonian, gpu):
    psi_0 = psi_deep(gpu)
    psi_0.params = (2 * np.random.rand(psi_0.num_params)) * psi_0.params
    psi = psi_deep(gpu)

    use_spins = ensemble.__name__.endswith("Spins")
    if not use_spins and psi_0.N % 3 != 0:
        return

    N = psi_0.N if use_spins else psi_0.N // 3

    ensemble = ensemble(N, gpu)

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
