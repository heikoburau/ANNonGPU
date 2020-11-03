from pyANNonGPU import PsiClassicalFP_1, PsiClassicalFP_2, PsiFullyPolarized
from QuantumExpression import PauliExpression
import numpy as np


def new_classical_network(
    num_sites,
    order,
    H_local,
    M_2=PauliExpression(1),
    M_1_squared=PauliExpression(1),
    params=0,
    prefactor=1,
    psi_ref="fully polarized",
    gpu=False
):
    assert order in (1, 2)
    assert psi_ref == "fully polarized"

    if order == 1:
        num_params = len(H_local)
        if params == 0:
            params = np.zeros(num_params, dtype=complex)

        return PsiClassicalFP_1(
            num_sites, H_local, M_2, M_1_squared, params, PsiFullyPolarized(num_sites), prefactor, gpu
        )

    if order == 2:
        num_params = len(H_local) + len(M_2) + len(M_1_squared) * (len(M_1_squared) + 1) // 2
        if params == 0:
            params = np.zeros(num_params, dtype=complex)

        return PsiClassicalFP_2(
            num_sites, H_local, M_2, M_1_squared, params, PsiFullyPolarized(num_sites), prefactor, gpu
        )


def new_2nd_order_vCN_from_H_local(
    num_sites,
    H_local_fun,
    distance,
    optimize_for_string_basis=True,
    params=0,
    prefactor=1,
    psi_ref="fully polarized",
    gpu=False
):
    H_local = H_local_fun(0)
    H_local.assign(1)

    H = sum(H_local_fun(l) for l in range(distance))
    H.assign(1)
    M_2 = (H**2).translationally_invariant(distance)
    M_2.assign(1)
    M_2 -= M_2[PauliExpression(1).pauli_string]

    M_1_squared = +H
    if optimize_for_string_basis:
        M_1_squared_pure_x = sum(m_1 for m_1 in M_1_squared if all(s[1] == 1 for s in m_1.pauli_string))
        M_1_squared_other = sum(m_1 for m_1 in M_1_squared if not all(s[1] == 1 for s in m_1.pauli_string))

        M_1_squared = M_1_squared_pure_x.translationally_invariant(distance) + M_1_squared_other

    M_1_squared -= M_1_squared[PauliExpression(1).pauli_string]

    print("H_local:", H_local)
    print("M_2:", M_2)
    print("M_1_squared:", M_1_squared)
