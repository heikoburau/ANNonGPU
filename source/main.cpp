#include "quantum_states.hpp"
#include "ensembles/MonteCarlo.hpp"
#include "operator/Operator.hpp"
#include "network_functions/HilbertSpaceDistance.hpp"
#include "types.h"

#include "QuantumExpression/QuantumExpression.hpp"

#include <cuda_profiler_api.h>

#include <iostream>
#include <complex>
#include <vector>


using namespace std;
using namespace ann_on_gpu;
using namespace quantum_expression;
using cpx = complex<double>;
using imap = map<int, int>;

const unsigned int N = 128;
const unsigned int M = 128;
const bool gpu = true;

int main(int argc, char *argv[]) {
    PauliExpression H(cpx(0.0, 0.0));

    for(auto i = 0; i < (int)N; i++) {
        H += PauliExpression(imap{{i, 3}, {(i + 1) % N, 3}});
        H += PauliExpression(imap{{i, 1}});
    }
    const auto U = cpx(0.0, -1.0) * H * 0.1;

    Operator op(U, gpu);

    PsiDeep psi(N, M, gpu);

    HilbertSpaceDistance hs_distance(N, psi.num_params, gpu);

    // MonteCarloSpins mc_loop(
    //     1u << 14u,
    //     1u,
    //     2u,
    //     gpu ? (1u << 10u) : 1u,
    //     gpu
    // );

    vector<cpx> result(psi.num_params);

    cudaProfilerStart();

    // cout << hs_distance.gradient(result.data(), psi, psi, op, false, mc_loop, 0.0) << endl;

    cudaProfilerStop();

    return 0;
}
