// ***********************************************************
// *       This is an automatically generated file.          *
// *       For editing, please use the source file:          *
// PsiOkVector.cu.template
// ***********************************************************

#include "network_functions/PsiOkVector.hpp"
#include "quantum_states.hpp"
#include "ensembles/ExactSummation.hpp"


namespace ann_on_gpu {

template<typename Psi_t, typename Ensemble>
Array<complex_t> psi_O_k_vector(Psi_t& psi, Ensemble& ensemble) {
    Array<complex_t> result(psi.num_params, psi.gpu);

    auto result_ptr = result.data();
    auto psi_kernel = psi.kernel();

    ensemble.foreach(
        psi,
        [=] __device__ __host__ (
            const unsigned int index,
            const typename Ensemble::Basis_t& configuration,
            const typename Psi_t::dtype log_psi,
            typename Psi_t::Payload& payload,
            const typename Psi_t::real_dtype weight
        ) {
            psi_kernel.foreach_O_k(
                configuration,
                payload,
                [&](const unsigned int k, const complex_t& O_k_element) {
                    result_ptr[k] = O_k_element;
                }
            );
        }
    );

    result.update_host();

    return result;
}

template<typename Psi_t, typename Basis_t>
Array<complex_t> psi_O_k(Psi_t& psi, const Basis_t& configuration) {
    Array<complex_t> result(psi.num_params, psi.gpu);

    auto result_ptr = result.data();
    auto psi_kernel = psi.kernel();
    auto conf = configuration;

    const auto functor = [=] __host__ __device__ () {
        #include "cuda_kernel_defines.h"

        SHARED typename Psi_t::Payload payload;
        psi_kernel.init_payload(payload, conf);

        psi_kernel.foreach_O_k(
            conf,
            payload,
            [&](const unsigned int k, const typename Psi_t::dtype& O_k_element) {
                result_ptr[k] = O_k_element;
            }
        );
    };

    if(psi.gpu) {
        cuda_kernel<<<1, psi.get_width()>>>(functor);
    }
    else {
        functor();
    }

    result.update_host();

    return result;
}


#if defined(ENABLE_SPINS)

template Array<complex_t> psi_O_k_vector(PsiDeep&, ExactSummation_t<Spins>&);
template Array<complex_t> psi_O_k(PsiDeep&, const Spins&);
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)

template Array<complex_t> psi_O_k_vector(PsiFullyPolarized&, ExactSummation_t<Spins>&);
template Array<complex_t> psi_O_k(PsiFullyPolarized&, const Spins&);
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)

template Array<complex_t> psi_O_k_vector(PsiClassicalFP<1u>&, ExactSummation_t<Spins>&);
template Array<complex_t> psi_O_k(PsiClassicalFP<1u>&, const Spins&);
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)

template Array<complex_t> psi_O_k_vector(PsiClassicalFP<2u>&, ExactSummation_t<Spins>&);
template Array<complex_t> psi_O_k(PsiClassicalFP<2u>&, const Spins&);
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)

template Array<complex_t> psi_O_k_vector(PsiClassicalANN<1u>&, ExactSummation_t<Spins>&);
template Array<complex_t> psi_O_k(PsiClassicalANN<1u>&, const Spins&);
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)

template Array<complex_t> psi_O_k_vector(PsiClassicalANN<2u>&, ExactSummation_t<Spins>&);
template Array<complex_t> psi_O_k(PsiClassicalANN<2u>&, const Spins&);
#endif
#if defined(ENABLE_PAULIS)

template Array<complex_t> psi_O_k_vector(PsiDeep&, ExactSummation_t<PauliString>&);
template Array<complex_t> psi_O_k(PsiDeep&, const PauliString&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)

template Array<complex_t> psi_O_k_vector(PsiFullyPolarized&, ExactSummation_t<PauliString>&);
template Array<complex_t> psi_O_k(PsiFullyPolarized&, const PauliString&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)

template Array<complex_t> psi_O_k_vector(PsiClassicalFP<1u>&, ExactSummation_t<PauliString>&);
template Array<complex_t> psi_O_k(PsiClassicalFP<1u>&, const PauliString&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)

template Array<complex_t> psi_O_k_vector(PsiClassicalFP<2u>&, ExactSummation_t<PauliString>&);
template Array<complex_t> psi_O_k(PsiClassicalFP<2u>&, const PauliString&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)

template Array<complex_t> psi_O_k_vector(PsiClassicalANN<1u>&, ExactSummation_t<PauliString>&);
template Array<complex_t> psi_O_k(PsiClassicalANN<1u>&, const PauliString&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)

template Array<complex_t> psi_O_k_vector(PsiClassicalANN<2u>&, ExactSummation_t<PauliString>&);
template Array<complex_t> psi_O_k(PsiClassicalANN<2u>&, const PauliString&);
#endif

} // namespace ann_on_gpu
