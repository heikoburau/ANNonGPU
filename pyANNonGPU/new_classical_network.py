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

    H_local = +H_local
    H_local.assign(1)

    log_prefactor = np.log(1 / 4**(num_sites / 2) if use_super_operator else 1 / 2**(num_sites / 2))

    if order == 1:
        num_params = len(H_local)
        if params == 0:
            params = np.zeros(num_params, dtype=complex)

        if use_super_operator:
            H_local = [pyANNonGPU.SuperOperator.from_expr(h, gpu) for h in H_local]
            H_2_local = [pyANNonGPU.SuperOperator.from_expr(PauliExpression(1), gpu)]
        else:
            H_local = [pyANNonGPU.Operator(h, gpu) for h in H_local]
            H_2_local = [pyANNonGPU.Operator(PauliExpression(1), gpu)]

        if psi_ref == "fully polarized":
            psi_ref = pyANNonGPU.PsiFullyPolarized(num_sites, log_prefactor)
            return pyANNonGPU.PsiClassicalFP_1(
                num_sites, H_local, H_2_local, params, psi_ref, log_prefactor, gpu
            )

        return pyANNonGPU.PsiClassicalANN_1(
            num_sites, H_local, H_2_local, params, psi_ref, log_prefactor, gpu
        )

    if order == 2:
        if distance == "max":
            distance = num_sites // 2

        if symmetric:
            H = sum(H_local.roll(l, num_sites) for l in range(distance))
            H_2_local = (H**2).translationally_invariant(distance)
            H_2_local.assign(1)
            H_2_local -= H_2_local[PauliExpression(1).pauli_string]
        else:
            H_2_local = H_local**2
            H_2_local.assign(1)
            H_2_local -= H_2_local[PauliExpression(1).pauli_string]

        H_local.assign(1)

        if symmetric:
            num_params = (
                len(H_local) +
                len(H_2_local) +
                distance * len(H_local) * (len(H_local) + 1) // 2
            )
        else:
            num_params = (
                len(H_local) +
                len(H_2_local) +
                len(H_local) * (len(H_local) + 1) // 2
            )

        if params == 0:
            params = np.zeros(num_params, dtype=complex)

        if use_super_operator:
            H_local = [pyANNonGPU.SuperOperator.from_expr(h, gpu) for h in H_local]
            H_2_local = [pyANNonGPU.SuperOperator.from_expr(h, gpu) for h in H_2_local]
        else:
            H_local = [pyANNonGPU.Operator(h, gpu) for h in H_local]
            H_2_local = [pyANNonGPU.Operator(h, gpu) for h in H_2_local]

        if psi_ref == "fully polarized":
            psi_ref = pyANNonGPU.PsiFullyPolarized(num_sites, log_prefactor)
            return pyANNonGPU.PsiClassicalFP_2(
                num_sites, H_local, H_2_local, params, psi_ref, log_prefactor, gpu
            )

        assert num_sites == psi_ref.num_sites

        return pyANNonGPU.PsiClassicalANN_2(
            num_sites, H_local, H_2_local, params, psi_ref, log_prefactor, gpu
        )
