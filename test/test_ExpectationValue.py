from pyANNonGPU import ExpectationValue, Operator
from pytest import approx
import numpy as np


def test_expectation_value(psi_deep, ensemble, all_operators, gpu):

    psi = psi_deep(gpu)

    use_spins = ensemble.__name__.endswith("Spins")
    if not use_spins and psi.N != 3 * psi.num_sites:
        return

    basis_name = "spins" if use_spins else "paulis"

    num_sites = psi.num_sites

    ensemble = ensemble(num_sites, gpu)

    H = all_operators(num_sites)
    H_op = Operator(H, gpu)
    H_matrix = H.matrix(num_sites, basis_name)

    psi.normalize(ensemble)

    e_value = ExpectationValue(gpu)

    state = psi.vector(ensemble)

    assert e_value(H_op, psi, ensemble) == approx(np.vdot(state, H_matrix @ state))
