from pyANNonGPU import apply_operator, Operator
import numpy as np
# from pathlib import Path


def test_operator(psi_all, all_operators, ensemble, gpu):
    psi = psi_all(gpu)

    num_sites = psi.num_sites

    ensemble = ensemble(num_sites, gpu)
    if ensemble.__class__.__name__.startswith("ExactSummation"):
        psi.normalize(ensemble)

    expr = all_operators(num_sites)
    op = Operator(expr, psi.gpu)

    vector = psi.vector(ensemble)
    test_vector = apply_operator(psi, op, ensemble)

    ref_vector = expr @ vector

    # np.save(Path.home() / "ref_vector.npy", ref_vector)
    # np.save(Path.home() / "test_vector.npy", test_vector)

    assert np.allclose(ref_vector, test_vector)
