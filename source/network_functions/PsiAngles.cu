#include "network_functions/PsiAngles.hpp"
#include "quantum_states.hpp"
#include "ensembles.hpp"
#include "basis/Spins.h"
#include "types.h"

namespace ann_on_gpu {


template<typename Psi_t, typename Ensemble>
Array<complex_t> psi_angles(const Psi_t& psi, Ensemble& ensemble) {
    Array<complex_t> result(psi.get_num_angles() * ensemble.get_num_steps(), ensemble.gpu);

    result.clear();

    auto psi_kernel = psi.kernel();
    auto result_data = result.data();

    ensemble.foreach(
        psi,
        [=] __device__ __host__ (
            const unsigned int index,
            const typename Ensemble::Basis_t& basis_vector,
            const typename Psi_t::dtype log_psi,
            typename Psi_t::dtype* angles,
            typename Psi_t::dtype* activations,
            const typename Psi_t::real_dtype weight
        ) {
            MULTI(j, psi_kernel.get_num_angles()) {
                result_data[index * psi_kernel.get_num_angles() + j] = angles[j];
            }
        }
    );

    result.update_host();

    return result;
}


#ifdef ENABLE_MONTE_CARLO

#ifdef ENABLE_SPINS
template Array<complex_t> psi_angles(const PsiDeep& psi, MonteCarloSpins& ensemble);
#endif // ENABLE_SPINS

#ifdef ENABLE_PAULIS
template Array<complex_t> psi_angles(const PsiDeep& psi, MonteCarloPaulis& ensemble);
#endif // ENABLE_PAULIS

#endif // ENABLE_MONTE_CARLO



#ifdef ENABLE_EXACT_SUMMATION

#ifdef ENABLE_SPINS
template Array<complex_t> psi_angles(const PsiDeep& psi, ExactSummationSpins& ensemble);
#endif // ENABLE_SPINS

#ifdef ENABLE_PAULIS
template Array<complex_t> psi_angles(const PsiDeep& psi, ExactSummationPaulis& ensemble);
#endif // ENABLE_PAULIS

#endif // ENABLE_EXACT_SUMMATION



} // namespace ann_on_gpu
