from pyANNonGPU import apply_operator, Operator
from pytest import approx


def test_operator(psi_all, all_operators, ensemble, gpu):
    psi = psi_all(gpu)

    num_sites = psi.num_sites

    ensemble = ensemble(num_sites, gpu)
    if ensemble.__class__.__name__.startswith("ExactSummation"):
        psi.normalize(ensemble)

    expr = all_operators(num_sites)
    op = Operator(expr, psi.gpu)

    vector = psi.vector(ensemble)
    vector_prime = apply_operator(psi, op, ensemble)

    assert approx(expr @ vector, vector_prime)
