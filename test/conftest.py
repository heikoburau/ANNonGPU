from pyANNonGPU import new_deep_neural_network, ExactSummationSpins
from QuantumExpression import sigma_x, sigma_y, sigma_z
import quantum_tools as qt


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", help="run tests also on GPU")


def pytest_generate_tests(metafunc):
    if 'gpu' in metafunc.fixturenames:
        if metafunc.config.getoption('gpu'):
            metafunc.parametrize("gpu", [True, False])
        else:
            metafunc.parametrize("gpu", [False])

    if 'mc' in metafunc.fixturenames:
        metafunc.parametrize("mc", [True, False])

    if 'psi_deep' in metafunc.fixturenames:
        psi_list = [
            lambda gpu: new_deep_neural_network(2, [2], [2], a=0.1, gpu=gpu),
            lambda gpu: new_deep_neural_network(3, [9, 6], [1, 3], noise=1e-2, gpu=gpu),
            lambda gpu: new_deep_neural_network(8, [16, 8, 4], [4, 2, 4], a=0.1, noise=1e-2, gpu=gpu),
        ]
        metafunc.parametrize("psi_deep", psi_list)

    if 'hamiltonian' in metafunc.fixturenames:
        metafunc.parametrize(
            "hamiltonian",
            [
                lambda L: qt.disordered_Heisenberg_chain(L, 1, 0.2, 1)
            ]
        )

    if 'single_sigma' in metafunc.fixturenames:
        metafunc.parametrize(
            "single_sigma",
            [
                lambda L: sigma_x(L // 2),
                lambda L: sigma_y(L // 2),
                lambda L: sigma_z(L // 2)
            ]
        )

    if 'ensemble' in metafunc.fixturenames:
        metafunc.parametrize(
            "ensemble",
            [
                lambda L, gpu: ExactSummationSpins(L, gpu)
            ]
        )
