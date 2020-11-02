// ***********************************************************
// *       This is an automatically generated file.          *
// *       For editing, please use the source file:          *
// PsiAngles.cu.template
// ***********************************************************

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
            const typename Ensemble::Basis_t& configuration,
            const typename Psi_t::dtype log_psi,
            typename Psi_t::Payload& payload,
            const typename Psi_t::real_dtype weight
        ) {
            MULTI(j, psi_kernel.get_num_angles()) {
                result_data[index * psi_kernel.get_num_angles() + j] = payload.angles[j];
            }
        }
    );

    result.update_host();

    return result;
}


#if defined(ENABLE_SPINS) && defined(ENABLE_MONTE_CARLO)
template Array<complex_t> psi_angles(const PsiDeep& psi, MonteCarlo_tt<Spins>& ensemble);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_MONTE_CARLO)
template Array<complex_t> psi_angles(const PsiDeep& psi, MonteCarlo_tt<PauliString>& ensemble);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS)
template Array<complex_t> psi_angles(const PsiDeep& psi, ExactSummation_t<Spins>& ensemble);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_EXACT_SUMMATION)
template Array<complex_t> psi_angles(const PsiDeep& psi, ExactSummation_t<PauliString>& ensemble);
#endif


} // namespace ann_on_gpu
