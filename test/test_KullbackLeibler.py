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

    psi.calibrate(esum)

    psi.params = 0.1 * complex_noise(len(psi.params))
    psi.normalize(esum)

    kl_divergence = KullbackLeibler(psi.psi_ref.num_params, gpu)
    div_test = kl_divergence(psi, psi.psi_ref, esum)

    psi_vec = psi.vector(esum)
    psi1_vec = psi.psi_ref.vector(esum)
    p = abs(psi_vec)**2

    assert np.sum(p) == approx(1)

    log_psi_vec = np.log(psi_vec)
    log_psi1_vec = np.log(psi1_vec)

    div_ref = (
        p @ abs(log_psi1_vec - log_psi_vec)**2 -
        abs(p @ (log_psi1_vec - log_psi_vec))**2
    )**0.5

    assert div_test == approx(div_ref)


def test_gradient(psi_classical_ann, gpu):
    psi = psi_classical_ann(gpu)

    esum = ExactSummationSpins(psi.num_sites, gpu)

    psi.calibrate(esum)

    psi.params = 0.1 * complex_noise(len(psi.params))
    psi.normalize(esum)
    psi_ann = psi.psi_ref

    # psi_ann = new_deep_neural_network(8, 8, [8, 4], [4, 2], noise=1e-3, a=0., gpu=gpu)

    assert len(psi_ann.params) == psi_ann.num_params

    kl_divergence = KullbackLeibler(psi_ann.num_params, gpu)

    gradient_test, _ = kl_divergence.gradient(psi, psi_ann, esum, 1)
    params_0 = psi_ann.params

    eps = 1e-4

    def distance_diff(delta_params):
        psi_ann.params = params_0 + delta_params
        plus_distance = kl_divergence(psi, psi_ann, esum)

        psi_ann.params = params_0 - delta_params
        minus_distance = kl_divergence(psi, psi_ann, esum)

        return (plus_distance - minus_distance) / (2 * eps)

    gradient_ref = np.zeros(psi_ann.num_params, dtype=complex)

    for k in range(psi_ann.num_params):
        delta_params = np.zeros_like(params_0)
        delta_params[k] = eps
        gradient_ref[k] = distance_diff(delta_params)

        delta_params = np.zeros_like(params_0)
        delta_params[k] = 1j * eps
        gradient_ref[k] += 1j * distance_diff(delta_params)

    # print(gradient_ref - gradient_test)
    # print(gradient_test)

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
