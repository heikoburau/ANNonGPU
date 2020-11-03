// ***********************************************************
// *       This is an automatically generated file.          *
// *       For editing, please use the source file:          *
// PsiVector.cu.template
// ***********************************************************

#ifdef ENABLE_EXACT_SUMMATION


#include "network_functions/PsiVector.hpp"
#include "quantum_states.hpp"
#include "ensembles/ExactSummation.hpp"
#include "types.h"

#include <iostream>

namespace ann_on_gpu {


template<typename Psi_t, typename Ensemble>
Array<complex_t> psi_vector(const Psi_t& psi, Ensemble& ensemble) {

    Array<complex_t> result(ensemble.get_num_steps(), ensemble.gpu);

    auto result_ptr = result.data();
    auto log_prefactor = log(psi.prefactor);
    auto psi_kernel = psi.kernel();

    ensemble.foreach(
        psi,
        [=] __host__ __device__ (
            const unsigned int conf_index,
            const typename Ensemble::Basis_t& configuration,
            const complex_t log_psi,
            typename Psi_t::Payload& payload,
            const double weight
        ) {
            #include "cuda_kernel_defines.h"

            SINGLE {
                result_ptr[conf_index] = exp(log_prefactor + log_psi);
            }
        }
    );
    result.update_host();

    return result;
}

template<typename Psi_t, typename Basis_t>
std::complex<double> log_psi_s(const Psi_t& psi, const Basis_t& configuration) {
    Array<complex_t> result(1u, psi.gpu);

    auto result_ptr = result.data();
    auto psi_kernel = psi.kernel();
    auto conf = configuration;

    const auto functor = [=] __host__ __device__ () {
        #include "cuda_kernel_defines.h"

        SHARED typename Psi_t::Payload payload;
        SHARED typename Psi_t::dtype   log_psi;

        psi_kernel.init_payload(payload, conf);
        psi_kernel.log_psi_s(log_psi, conf, payload);

        SINGLE {
            *result_ptr = log_psi;
        }
    };

    if(psi.gpu) {
        cuda_kernel<<<1, psi.get_width()>>>(functor);
    }
    else {
        functor();
    }

    result.update_host();

    return result.front().to_std();
}


#if defined(ENABLE_SPINS)
template std::complex<double> log_psi_s(const PsiDeep&, const Spins&);
template Array<complex_t> psi_vector(const PsiDeep& psi, ExactSummation_t<Spins>& ensemble);
#endif
#if defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_SPINS)
template std::complex<double> log_psi_s(const PsiFullyPolarized&, const Spins&);
template Array<complex_t> psi_vector(const PsiFullyPolarized& psi, ExactSummation_t<Spins>& ensemble);
#endif
#if defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_SPINS)
template std::complex<double> log_psi_s(const PsiClassicalFP<1u>&, const Spins&);
template Array<complex_t> psi_vector(const PsiClassicalFP<1u>& psi, ExactSummation_t<Spins>& ensemble);
#endif
#if defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_SPINS)
template std::complex<double> log_psi_s(const PsiClassicalFP<2u>&, const Spins&);
template Array<complex_t> psi_vector(const PsiClassicalFP<2u>& psi, ExactSummation_t<Spins>& ensemble);
#endif
#if defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL_ANN)
template std::complex<double> log_psi_s(const PsiClassicalANN<1u>&, const Spins&);
template Array<complex_t> psi_vector(const PsiClassicalANN<1u>& psi, ExactSummation_t<Spins>& ensemble);
#endif
#if defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL_ANN)
template std::complex<double> log_psi_s(const PsiClassicalANN<2u>&, const Spins&);
template Array<complex_t> psi_vector(const PsiClassicalANN<2u>& psi, ExactSummation_t<Spins>& ensemble);
#endif
#if defined(ENABLE_PAULIS)
template std::complex<double> log_psi_s(const PsiDeep&, const PauliString&);
template Array<complex_t> psi_vector(const PsiDeep& psi, ExactSummation_t<PauliString>& ensemble);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template std::complex<double> log_psi_s(const PsiFullyPolarized&, const PauliString&);
template Array<complex_t> psi_vector(const PsiFullyPolarized& psi, ExactSummation_t<PauliString>& ensemble);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template std::complex<double> log_psi_s(const PsiClassicalFP<1u>&, const PauliString&);
template Array<complex_t> psi_vector(const PsiClassicalFP<1u>& psi, ExactSummation_t<PauliString>& ensemble);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template std::complex<double> log_psi_s(const PsiClassicalFP<2u>&, const PauliString&);
template Array<complex_t> psi_vector(const PsiClassicalFP<2u>& psi, ExactSummation_t<PauliString>& ensemble);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template std::complex<double> log_psi_s(const PsiClassicalANN<1u>&, const PauliString&);
template Array<complex_t> psi_vector(const PsiClassicalANN<1u>& psi, ExactSummation_t<PauliString>& ensemble);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template std::complex<double> log_psi_s(const PsiClassicalANN<2u>&, const PauliString&);
template Array<complex_t> psi_vector(const PsiClassicalANN<2u>& psi, ExactSummation_t<PauliString>& ensemble);
#endif


} // namespace ann_on_gpu


#endif // ENABLE_EXACT_SUMMATION
