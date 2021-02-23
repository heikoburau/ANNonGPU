from ._pyANNonGPU import SuperOperator, SparseMatrix
from QuantumExpression import PauliExpression
from scipy.sparse import find
import numpy as np
from collections import Counter


@staticmethod
def from_expr(expr, part, U=None, gpu=False):
    matrices = []
    assert part in ("real", "imag")

    for term in expr:
        pauli_string = dict(term.pauli_string)
        indices = list(pauli_string)
        site_i = indices[0]
        site_j = indices[1] if len(indices) == 2 else 0

        coeff = term.coefficient

        assert len(pauli_string) in (1, 2)

        two_sites = len(pauli_string) == 2

        if two_sites:
            m = PauliExpression({0: pauli_string[site_i], 1: pauli_string[site_j]})
        else:
            m = PauliExpression({0: pauli_string[site_i]})

        m = m.sparse_matrix(len(pauli_string), "paulis", U)

        if part == "real":
            m = m.real
        if part == "imag":
            m = m.imag

        I, J, V = find(m)

        num_matrices = Counter(J).most_common(1)[0][1]

        values = [np.zeros(16, dtype=float) for n in range(num_matrices)]
        col_to_row = [np.zeros(16, dtype=int) for n in range(num_matrices)]

        for i, j, v in zip(I, J, V):
            n = next(n for n in range(num_matrices) if values[n][j] == 0)

            values[n][j] = coeff.real * v.real
            col_to_row[n][j] = i

        for n in range(num_matrices):
            matrices.append(SparseMatrix(
                two_sites,
                site_i,
                site_j,
                values[n],
                col_to_row[n]
            ))

    return SuperOperator(matrices, gpu)


setattr(SuperOperator, "from_expr", from_expr)
