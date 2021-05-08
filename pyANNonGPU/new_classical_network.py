import pyANNonGPU
from QuantumExpression import PauliExpression
import numpy as np


def new_classical_network(
    num_sites,
    order,
    H_local,
    symmetric=True,
    distance="max",
    params=0,
    psi_ref="fully polarized",
    use_super_operator=False,
    gpu=False
):
    assert order in (1, 2)

    log_prefactor = np.log(1 / 4**(num_sites / 2) if use_super_operator else 1 / 2**(num_sites / 2))

    if order == 1:
        num_params = len(H_local)
        if params == 0:
            params = np.zeros(num_params, dtype=complex)

        if use_super_operator:
            H_local = [pyANNonGPU.SuperOperator.from_expr(h, gpu) for h in H_local]
        else:
            H_local = [pyANNonGPU.Operator(h, gpu) for h in H_local]

        if psi_ref == "fully polarized":
            psi_ref = pyANNonGPU.PsiFullyPolarized(num_sites, log_prefactor)
            return pyANNonGPU.PsiClassicalFP_1(
                num_sites, H_local, params, psi_ref, log_prefactor, gpu
            )

        return pyANNonGPU.PsiClassicalANN_1(
            num_sites, H_local, params, psi_ref, 0, gpu
        )

    if order == 2:
        if symmetric:
            num_params = (
                len(H_local) +
                distance * len(H_local) * (len(H_local) + 1) // 2
            )
        else:
            num_params = (
                len(H_local) +
                len(H_local) * (len(H_local) + 1) // 2
            )

        if params == 0:
            params = np.zeros(num_params, dtype=complex)

        if use_super_operator:
            H_local = [pyANNonGPU.SuperOperator.from_expr(h, gpu) for h in H_local]
        else:
            H_local = [pyANNonGPU.Operator(h, gpu) for h in H_local]

        if psi_ref == "fully polarized":
            psi_ref = pyANNonGPU.PsiFullyPolarized(num_sites, log_prefactor)
            return pyANNonGPU.PsiClassicalFP_2(
                num_sites, H_local, params, psi_ref, log_prefactor, gpu
            )

        assert num_sites == psi_ref.num_sites

        return pyANNonGPU.PsiClassicalANN_2(
            num_sites, H_local, params, psi_ref, 0, gpu
        )
