from ._pyANNonGPU import SuperOperator
from QuantumExpression import sigma_x, sigma_y, sigma_z
import numpy as np
import math


U = np.array([
    [1, 0, 0, 1],
    [0, 1, 1j, 0],
    [0, 1, -1j, 0],
    [1, 0, 0, -1]
]) / math.sqrt(2)

Sx = U @ sigma_x(0).matrix(1, "paulis") @ U.T.conj()
Sy = U @ sigma_y(0).matrix(1, "paulis") @ U.T.conj()
Sz = U @ sigma_z(0).matrix(1, "paulis") @ U.T.conj()

op_map = {
    0: np.eye(4),
    1: Sx,
    2: Sy,
    3: Sz
}


@staticmethod
def from_expr(expr, gpu):
    coefficients = []
    site_indices = []
    matrices = []

    for term in expr:
        site_indices_row = []
        matrices_row = []
        for i, t in term.pauli_string:
            site_indices_row.append(i)
            matrices_row.append(op_map[t])

        coefficients.append(term.coefficient)
        site_indices.append(site_indices_row)
        matrices.append(matrices_row)

    return SuperOperator(coefficients, site_indices, matrices, gpu)


setattr(SuperOperator, "from_expr", from_expr)
