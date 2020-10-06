#include "network_functions/PsiAngles.hpp"
#include "quantum_states.hpp"
#include "ensembles.hpp"
#include "operator/Spins.h"
#include "types.h"

namespace ann_on_gpu {


template<typename Psi_t, typename SpinEnsemble>
Array<complex_t> psi_angles(const Psi_t& psi, SpinEnsemble& spin_ensemble) {
    Array<complex_t> result(psi.get_num_angles() * spin_ensemble.get_num_steps(), spin_ensemble.gpu);

    result.clear();

    auto psi_kernel = psi.kernel();
    auto result_data = result.data();

    spin_ensemble.foreach(
        psi,
        [=] __device__ __host__ (
            const unsigned int spin_index,
            const Spins spins,
            const typename Psi_t::dtype log_psi,
            typename Psi_t::dtype* angles,
            typename Psi_t::dtype* activations,
            const typename Psi_t::real_dtype weight
        ) {
            MULTI(j, psi_kernel.get_num_angles()) {
                result_data[spin_index * psi_kernel.get_num_angles() + j] = angles[j];
            }
        }
    );

    result.update_host();

    return result;
}


#ifdef ENABLE_PSI_DEEP

#ifdef ENABLE_EXACT_SUMMATION
template Array<complex_t> psi_angles(const PsiDeep& psi, ExactSummation& spin_ensemble);
#endif // ENABLE_EXACT_SUMMATION

#ifdef ENABLE_MONTE_CARLO
template Array<complex_t> psi_angles(const PsiDeep& psi, MonteCarloLoop& spin_ensemble);
#endif // ENABLE_MONTE_CARLO

#endif // ENABLE_PSI_DEEP

} // namespace ann_on_gpu
