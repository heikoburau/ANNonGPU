import pyANNonGPU
from QuantumExpression import PauliExpression
import numpy as np


def new_classical_network(
    num_sites,
    order,
    H_local,
    H_local_diagonal=PauliExpression(),
    H_2_local=PauliExpression(1),
    distance="max",
    params=0,
    prefactor=1,
    psi_ref="fully polarized",
    gpu=False
):
    assert order in (1, 2)

    if order == 1:
        num_params = len(H_local)
        if params == 0:
            params = np.zeros(num_params, dtype=complex)

        if psi_ref == "fully polarized":
            psi_ref = pyANNonGPU.PsiFullyPolarized(num_sites)
            return pyANNonGPU.PsiClassicalFP_1(
                num_sites, H_local_diagonal, H_local, H_2_local, params, psi_ref, prefactor, gpu
            )

        return pyANNonGPU.PsiClassicalANN_1(
            num_sites, H_local_diagonal, H_local, H_2_local, params, psi_ref, prefactor, gpu
        )

    if order == 2:
        if distance == "max":
            distance = num_sites // 2

        num_params = (
            len(H_local_diagonal) +
            len(H_local) +
            len(H_2_local) +
            distance * len(H_local) * (len(H_local) + 1) // 2
        )
        if params == 0:
            params = np.zeros(num_params, dtype=complex)

        if psi_ref == "fully polarized":
            psi_ref = pyANNonGPU.PsiFullyPolarized(num_sites)
            return pyANNonGPU.PsiClassicalFP_2(
                num_sites, H_local_diagonal, H_local, H_2_local, params, psi_ref, prefactor, gpu
            )

        assert num_sites == psi_ref.num_sites

        return pyANNonGPU.PsiClassicalANN_2(
            num_sites, H_local_diagonal, H_local, H_2_local, params, psi_ref, prefactor, gpu
        )


def new_2nd_order_vCN_from_H_local(
    num_sites,
    H_local,
    H_local_diagonal=PauliExpression(),
    distance="max",
    params=0,
    prefactor=1,
    psi_ref="fully polarized",
    gpu=False
):
    if distance == "max":
        distance = num_sites // 2

    H = sum(H_local.roll(l, num_sites) for l in range(distance))
    H_2_local = (H**2).translationally_invariant(distance)
    H_2_local.assign(1)
    H_2_local -= H_2_local[PauliExpression(1).pauli_string]

    H_local.assign(1)

    return new_classical_network(
        num_sites, 2, H_local, H_local_diagonal, H_2_local, distance, params, prefactor, psi_ref, gpu
    )
