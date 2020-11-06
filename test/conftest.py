from pyANNonGPU import new_deep_neural_network, new_classical_network, new_2nd_order_vCN_from_H_local, ExactSummationSpins#, ExactSummationPaulis
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
            lambda gpu: new_deep_neural_network(2, 2, [2], [2], a=0.1, gpu=gpu),
            lambda gpu: new_deep_neural_network(3, 3, [9, 6], [1, 3], noise=1e-2, gpu=gpu),
            lambda gpu: new_deep_neural_network(2, 6, [12, 6], [6, 12], noise=1e-2, a=-0.2, gpu=gpu),
            lambda gpu: new_deep_neural_network(3, 9, [18, 9, 3], [2, 2, 3], a=0.1, noise=1e-3, gpu=gpu),
        ]
        metafunc.parametrize("psi_deep", psi_list)

    if 'psi_all' in metafunc.fixturenames:
        psi_list = [
            lambda gpu: new_classical_network(2, 1, sigma_z(0) * sigma_z(1) + sigma_x(0), gpu=gpu),
            lambda gpu: new_classical_network(6, 1, sigma_z(0) * sigma_z(1) + sigma_x(0) + sigma_x(0) * sigma_x(1), gpu=gpu),
            lambda gpu: new_classical_network(4, 1, sigma_z(0) * sigma_z(1) + sigma_x(0), gpu=gpu),
            lambda gpu: new_deep_neural_network(2, 2, [2], [2], a=0.1, gpu=gpu),
            lambda gpu: new_deep_neural_network(3, 3, [9, 6], [1, 3], noise=1e-2, gpu=gpu),
            lambda gpu: new_deep_neural_network(2, 6, [12, 6], [6, 12], noise=1e-2, a=-0.2, gpu=gpu),
            lambda gpu: new_deep_neural_network(3, 9, [18, 9, 3], [2, 2, 3], a=0.1, noise=1e-3, gpu=gpu),
            lambda gpu: new_2nd_order_vCN_from_H_local(
                6, lambda l: sigma_z(l) * sigma_z((l + 1) % 6) + 1.1 * sigma_y(l), gpu=gpu
            ),
        ]
        metafunc.parametrize("psi_all", psi_list)

    if 'psi_classical' in metafunc.fixturenames:
        psi_list = [
            lambda gpu: new_classical_network(4, 1, sigma_z(0) * sigma_z(1) + sigma_x(0), gpu=gpu),
            lambda gpu: new_classical_network(6, 1, sigma_z(0) * sigma_x(1) * sigma_z(2) + sigma_y(0) + sigma_x(0) * sigma_x(1), gpu=gpu),
            lambda gpu: new_classical_network(4, 1, sigma_z(0) * sigma_z(1) + sigma_x(0), gpu=gpu),
            lambda gpu: new_2nd_order_vCN_from_H_local(2, lambda l: sigma_z(l) * sigma_z((l + 1) % 2) + 1.1 * sigma_x(l), gpu=gpu),
            lambda gpu: new_2nd_order_vCN_from_H_local(4, lambda l: sigma_z(l) * sigma_y((l + 1) % 2) + 1.1 * sigma_x(l), gpu=gpu),
            lambda gpu: new_2nd_order_vCN_from_H_local(
                4, lambda l: sigma_z(l) * sigma_y((l + 1) % 2) + 1.1 * sigma_x(l),
                psi_ref=new_deep_neural_network(4, 4, [4], [4], noise=1e-2, a=0.1, gpu=gpu),
                gpu=gpu
            ),
            lambda gpu: new_classical_network(
                8, 1, sigma_z(0) * sigma_x(1) * sigma_z(2) + 1.2 * sigma_y(0) + 1.1 * sigma_x(0) * sigma_x(1),
                psi_ref=new_deep_neural_network(8, 8, [8], [8], noise=1e-2, a=0., gpu=gpu),
                gpu=gpu
            )
        ]
        metafunc.parametrize("psi_classical", psi_list)

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
                ExactSummationSpins,
                # ExactSummationPaulis
            ]
        )
