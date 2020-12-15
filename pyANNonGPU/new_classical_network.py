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
    U_matrix=None,
    gpu=False
):
    assert order in (1, 2)

    H_local = list(H_local)
    for h in H_local:
        h.assign(1)

    prefactor = 1 / 4**(num_sites / 2) if use_super_operator else 1 / 2**(num_sites / 2)

    if order == 1:
        num_params = len(H_local)
        if params == 0:
            params = np.zeros(num_params, dtype=complex)

        if use_super_operator:
            H_local = [pyANNonGPU.SuperOperator.from_expr(h, U_matrix, gpu) for h in H_local]
            # H_2_local = [pyANNonGPU.SuperOperator.from_expr(PauliExpression(1), U_matrix, gpu)]
            H_2_local = []
        else:
            H_local = [pyANNonGPU.Operator(h, gpu) for h in H_local]
            H_2_local = [pyANNonGPU.Operator(PauliExpression(1), gpu)]

        if psi_ref == "fully polarized":
            psi_ref = pyANNonGPU.PsiFullyPolarized(num_sites, prefactor)
            return pyANNonGPU.PsiClassicalFP_1(
                num_sites, H_local, H_2_local, params, psi_ref, prefactor, gpu
            )

        return pyANNonGPU.PsiClassicalANN_1(
            num_sites, H_local, H_2_local, params, psi_ref, prefactor, gpu
        )

    if order == 2:
        if distance == "max":
            distance = num_sites // 2

        if symmetric:
            H = sum(H_local.roll(l, num_sites) for l in range(distance))
            H_2_local = (H**2).translationally_invariant(distance)
            H_2_local.assign(1)
            H_2_local -= H_2_local[PauliExpression(1).pauli_string]

            num_params = (
                len(H_local) +
                len(H_2_local) +
                distance * len(H_local) * (len(H_local) + 1) // 2
            )
        else:
            cell_size = len(H_local) // num_sites

            H_2_local = []
            for i in range(len(H_local)):
                cell_i = i // cell_size

                for cell_j in range(cell_i, cell_i + distance):
                    cell_j_offset = (cell_j % num_sites) * cell_size
                    for j in range(cell_j_offset, cell_j_offset + cell_size):
                        H_ij = H_local[i] * H_local[j]
                        H_ij.assign(1)
                        H_ij -= H_ij[PauliExpression(1).pauli_string]
                        H_ij = H_ij.crop(1e-2)

                        if not H_ij.is_numeric:
                            H_2_local.append(H_ij)

                        # print(H_ij)

            num_pairs = len(H_local) * distance * cell_size

            num_params = (
                len(H_local) +
                len(H_2_local) +
                num_pairs
            )

        if params == 0:
            params = np.zeros(num_params, dtype=complex)

        if use_super_operator:
            H_local = [pyANNonGPU.SuperOperator.from_expr(h, U_matrix, gpu) for h in H_local]
            H_2_local = [pyANNonGPU.SuperOperator.from_expr(h, U_matrix, gpu) for h in H_2_local]
        else:
            H_local = [pyANNonGPU.Operator(h, gpu) for h in H_local]
            H_2_local = [pyANNonGPU.Operator(h, gpu) for h in H_2_local]

        if psi_ref == "fully polarized":
            psi_ref = pyANNonGPU.PsiFullyPolarized(num_sites, prefactor)
            return pyANNonGPU.PsiClassicalFP_2(
                num_sites, H_local, H_2_local, params, psi_ref, prefactor, gpu
            )

        assert num_sites == psi_ref.num_sites

        return pyANNonGPU.PsiClassicalANN_2(
            num_sites, H_local, H_2_local, params, psi_ref, prefactor, gpu
        )
