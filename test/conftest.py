from pyANNonGPU import (
    # new_deep_neural_network,
    new_convolutional_network,
    # new_classical_network,
    ExactSummationSpins
)
from QuantumExpression import sigma_x, sigma_y, sigma_z
import quantum_tools as qt


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", help="run tests also on GPU")


def pytest_generate_tests(metafunc):
    if 'gpu' in metafunc.fixturenames:
        if metafunc.config.getoption('gpu'):
            metafunc.parametrize("gpu", [True])
        else:
            metafunc.parametrize("gpu", [False])

    if 'mc' in metafunc.fixturenames:
        metafunc.parametrize("mc", [True, False])

    if 'psi_deep' in metafunc.fixturenames:
        psi_list = [
            lambda gpu: new_deep_neural_network(2, 2, [2], [2], a=0, gpu=gpu),
            # lambda gpu: new_deep_neural_network(3, 3, [9, 6], [1, 3], noise=1e-2, gpu=gpu),
            # lambda gpu: new_deep_neural_network(2, 6, [12, 6], [6, 12], noise=1e-2, a=-0.2, gpu=gpu),
            lambda gpu: new_deep_neural_network(3, 3, [18, 9, 3], [3, 2, 3], a=0, noise=1e-3, gpu=gpu),
            # lambda gpu: new_deep_neural_network(8, 8, [8], [8], noise=1e-2, a=0., gpu=gpu),
            # lambda gpu: new_deep_neural_network(8, 8, [8, 8], [4, 4], noise=1e-3, a=0., gpu=gpu)
        ]
        metafunc.parametrize("psi_deep", psi_list)

    if 'psi_cnn' in metafunc.fixturenames:
        psi_list = [
            lambda gpu: new_convolutional_network([3], [(1, [3])], noise=1e-1, gpu=gpu),
            lambda gpu: new_convolutional_network([5], [(3, [3])], noise=1e-1, gpu=gpu),
        ]
        metafunc.parametrize("psi_cnn", psi_list)

    if 'psi_all' in metafunc.fixturenames:
        psi_list = [
            # lambda gpu: new_classical_network(2, 1, sigma_z(0) * sigma_z(1) + sigma_x(0), gpu=gpu),
            # lambda gpu: new_classical_network(6, 1, sigma_z(0) * sigma_z(1) + sigma_x(0) + sigma_x(0) * sigma_x(1), gpu=gpu),
            # lambda gpu: new_classical_network(4, 1, sigma_z(0) * sigma_z(1) + sigma_x(0), gpu=gpu),
            # lambda gpu: new_deep_neural_network(3, 3, [9, 6], [3, 3], noise=1e-2, gpu=gpu),
            # lambda gpu: new_deep_neural_network(2, 6, [12, 6], [6, 12], noise=1e-2, a=-0.2, gpu=gpu),
            # lambda gpu: new_deep_neural_network(3, 9, [18, 9, 3], [2, 2, 3], a=0.1, noise=1e-3, gpu=gpu),
            # lambda gpu: new_convolutional_network([3], [(1, [3])], noise=1e-1, gpu=gpu),
            # lambda gpu: new_convolutional_network([5], [(3, [3])], noise=1e-1, gpu=gpu),
            # lambda gpu: new_convolutional_network([2], [(2, [2]), (1, [2])], noise=1e-1, gpu=gpu)
            # lambda gpu: new_convolutional_network([5], [(3, [3]), (4, [5])], noise=1e-2, gpu=gpu)
            # lambda gpu: new_classical_network(
            #     6, 2, sigma_z(0) * sigma_z((0 + 1) % 6) + 1.1 * sigma_y(0), gpu=gpu
            # ),
            # lambda gpu: new_classical_network(
            #     4, 2, [sigma_z(0) * sigma_x(1) * sigma_z(2), 1.2 * sigma_y(0) + 1.1 * sigma_x(0) * sigma_x(1)],
            #     symmetric=False,
            #     psi_ref=new_deep_neural_network(4, 4, [4], [4], noise=1e-2, gpu=gpu),
            #     gpu=gpu
            # )
        ]
        metafunc.parametrize("psi_all", psi_list)

    if 'psi_classical' in metafunc.fixturenames:
        psi_list = [
            lambda gpu: new_classical_network(4, 1, sigma_z(0) * sigma_z(1) + sigma_x(0), gpu=gpu),
            lambda gpu: new_classical_network(6, 1, sigma_z(0) * sigma_x(1) * sigma_z(2) + sigma_y(0) + sigma_x(0) * sigma_x(1), gpu=gpu),
            lambda gpu: new_classical_network(4, 1, sigma_z(0) * sigma_z(1) + sigma_x(0), gpu=gpu),
            # lambda gpu: new_classical_network(2, 2, sigma_z(0) * sigma_z((0 + 1) % 2) + 1.1 * sigma_x(0), gpu=gpu),
            # lambda gpu: new_classical_network(4, 2, sigma_z(0) * sigma_y((0 + 1) % 2) + 1.1 * sigma_x(0), gpu=gpu),
            # lambda gpu: new_classical_network(
            #     4, 2, sigma_z(0) * sigma_z(1) + 1.1 * sigma_x(0),
            #     psi_ref=new_deep_neural_network(4, 4, [4], [4], noise=1e-2, a=0.2, gpu=gpu),
            #     gpu=gpu
            # ),
            # lambda gpu: new_classical_network(
            #     8, 1, sigma_z(0) * sigma_x(1) * sigma_z(2) + 1.2 * sigma_y(0) + 1.1 * sigma_x(0) * sigma_x(1),
            #     psi_ref=new_deep_neural_network(8, 8, [8], [8], noise=1e-2, a=0., gpu=gpu),
            #     gpu=gpu
            # )
        ]
        metafunc.parametrize("psi_classical", psi_list)

    if 'psi_classical_ann' in metafunc.fixturenames:
        psi_list = [
            lambda gpu: new_classical_network(
                4, 2, sigma_z(0) * sigma_y(1) + 1.1 * sigma_x(0),
                psi_ref=new_deep_neural_network(4, 4, [4], [4], noise=1e-2, a=0.1, gpu=gpu),
                gpu=gpu
            ),
            lambda gpu: new_classical_network(
                8, 1, sigma_z(0) * sigma_x(1) * sigma_z(2) + 1.2 * sigma_y(0) + 1.1 * sigma_x(0) * sigma_x(1),
                psi_ref=new_deep_neural_network(8, 8, [8], [8], noise=1e-3, a=0., gpu=gpu),
                gpu=gpu
            ),
            lambda gpu: new_classical_network(
                8, 1, sigma_z(0) * sigma_x(1) * sigma_z(2) + 1.2 * sigma_y(0) + 1.1 * sigma_x(0) * sigma_x(1),
                psi_ref=new_deep_neural_network(8, 8, [8], [8], noise=1e-3, a=0., gpu=gpu),
                gpu=gpu
            )
        ]
        metafunc.parametrize("psi_classical_ann", psi_list)

    if 'hamiltonian' in metafunc.fixturenames:
        metafunc.parametrize(
            "hamiltonian",
            [
                lambda L: qt.disordered_Heisenberg_chain(L, 1, 0.2, 1)
            ]
        )

    if 'all_operators' in metafunc.fixturenames:
        metafunc.parametrize(
            "all_operators",
            [
                lambda L: sigma_x(L // 2),
                lambda L: sigma_y(L // 2),
                lambda L: sigma_z(L // 2),
                lambda L: qt.disordered_Heisenberg_chain(L, 1, 0.2, 1),
                lambda L: qt.disordered_Heisenberg_chain(L // 2, 0.5, 0.3, 0.7)
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
                ExactSummationSpins,
                # ExactSummationPaulis
            ]
        )
