from pyANNonGPU import LearningByGradientDescent
from math import acos, sqrt


def two_state_probability(psi, phi):
    psi_dot_phi = psi.conj() @ phi
    return min(
        (psi_dot_phi * psi_dot_phi.conj() / ((psi.conj() @ psi) * (phi.conj() @ phi))).real,
        1.0
    )


def fubini_study(psi, phi):
    return acos(sqrt(two_state_probability(psi, phi)))


def test_operator(psi_deep, single_sigma, ensemble, gpu):
    psi_0 = psi_deep(gpu)

    use_spins = ensemble.__name__.endswith("Spins")
    if not use_spins and psi_0.N % 3 != 0:
        return

    L = psi_0.N if use_spins else psi_0.N // 3

    op = single_sigma(L)
    ensemble = ensemble(L, gpu)
    psi_0.normalize(ensemble)

    print("L:", L)
    print("operator:", op)

    learning = LearningByGradientDescent(psi_0.num_params, ensemble)

    U = 1 + 0.1j * op

    psi_out = learning.optimize_for(psi_0, psi_0, U, True, eta=1e-2)

    assert fubini_study(
        U.matrix(L, "spins" if use_spins else "paulis") @ psi_0.vector(ensemble),
        psi_out.vector(ensemble)
    ) < 1e-2

    U = 1 + 0.1 * op

    psi_out = learning.optimize_for(psi_0, psi_0, U, True, eta=1e-2)

    assert fubini_study(
        U.matrix(L, "spins" if use_spins else "paulis") @ psi_0.vector(ensemble),
        psi_out.vector(ensemble)
    ) < 1e-2
