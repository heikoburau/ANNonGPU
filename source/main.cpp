#include "quantum_states.hpp"
#include "ensembles/MonteCarlo.hpp"
// #include "ensembles/ExactSummation.hpp"
#include "operator/Operator.hpp"
// #include "network_functions/HilbertSpaceDistance.hpp"
#include "network_functions/ExpectationValue.hpp"
#include "types.h"

#include "QuantumExpression/QuantumExpression.hpp"

#include <cuda_profiler_api.h>

#include <map>
#include <iostream>
#include <complex>
#include <vector>
#include <chrono>


using namespace std;
using namespace std::chrono;
using namespace ann_on_gpu;
using namespace quantum_expression;
using cpx = complex<double>;
using imap = map<int, int>;

const unsigned int N = 14;
const unsigned int M = 14;
const bool gpu = true;

int main(int argc, char *argv[]) {
    PauliExpression H(cpx(0.0, 0.0));

    for(auto i = 0; i < (int)N; i++) {
        H += PauliExpression(imap{{i, 3}, {(i + 1) % N, 3}});
        H += PauliExpression(imap{{i, 2}, {(i + 1) % N, 2}});
        H += PauliExpression(imap{{i, 1}});
    }
    const auto U = cpx(0.0, -1.0) * H * 0.1;

    Operator op(U, gpu);

    PsiDeep psi(N, M, gpu);

    // HilbertSpaceDistance hs_distance(psi.num_params, gpu);
    ExpectationValue evalue(gpu);

    MonteCarloSpins mc_loop(
        1u << 12u,
        1u,
        2u,
        gpu ? (1u << 12u) : 1u,
        Update_Policy<ann_on_gpu::Spins>(),
        gpu
    );

    // ExactSummationSpins esum(N, gpu);

    // vector<cpx> result(psi.num_params);

    // cudaProfilerStart();

    const auto start = high_resolution_clock::now();

    cout << evalue(op, psi, mc_loop) << endl;

    const auto end = high_resolution_clock::now();

    // // cout << hs_distance.gradient(result.data(), psi, psi, op, false, mc_loop, 0.0) << endl;

    // cudaProfilerStop();

    duration<double, milli> fp_ms = end - start;

    cout << "duration: " << fp_ms.count() << endl;

    return 0;
}
