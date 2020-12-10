from ._pyANNonGPU import SuperOperator
from QuantumExpression import sigma_x, sigma_y, sigma_z
import numpy as np
# import math


# U = np.array([
#     [1, 0, 0, 1],
#     [0, 1, 1j, 0],
#     [0, 1, -1j, 0],
#     [1, 0, 0, -1]
# ]) / math.sqrt(2)


@staticmethod
def from_expr(expr, U, gpu):
    coefficients = []
    site_indices = []
    matrices = []

    num_sites = max(
        max(i for i, t in term.pauli_string) if term.pauli_string else 1
        for term in expr
    ) + 1

    if not isinstance(U, (list, tuple)):
        U = [U] * num_sites

    Sx = sigma_x(0).matrix(1, "paulis")
    Sy = sigma_y(0).matrix(1, "paulis")
    Sz = sigma_z(0).matrix(1, "paulis")

    op_map = [
        {
            0: np.eye(4),
            1: U[i] @ Sx @ U[i].T.conj(),
            2: U[i] @ Sy @ U[i].T.conj(),
            3: U[i] @ Sz @ U[i].T.conj(),
        } for i in range(num_sites)
    ]

    for term in expr:
        site_indices_row = []
        matrices_row = []
        for i, t in term.pauli_string:
            site_indices_row.append(i)
            matrices_row.append(op_map[i][t])

        coefficients.append(term.coefficient)
        site_indices.append(site_indices_row)
        matrices.append(matrices_row)

    return SuperOperator(coefficients, site_indices, matrices, gpu)


setattr(SuperOperator, "from_expr", from_expr)
