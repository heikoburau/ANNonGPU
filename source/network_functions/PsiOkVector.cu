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
Array<complex_t> psi_O_k_vector(const Psi_t& psi, Ensemble& ensemble) {
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
Array<complex_t> psi_O_k(const Psi_t& psi, const Basis_t& configuration) {
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
        psi_kernel.log_psi_s(log_psi, configuration, payload);

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
template Array<complex_t> psi_O_k_vector(const PsiDeep&, ExactSummation_t<Spins>&);
template Array<complex_t> psi_O_k(const PsiDeep&, const Spins&);
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template std::complex<double> log_psi_s(const PsiClassicalFP<1u>&, const Spins&);
template Array<complex_t> psi_O_k_vector(const PsiClassicalFP<1u>&, ExactSummation_t<Spins>&);
template Array<complex_t> psi_O_k(const PsiClassicalFP<1u>&, const Spins&);
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template std::complex<double> log_psi_s(const PsiClassicalFP<2u>&, const Spins&);
template Array<complex_t> psi_O_k_vector(const PsiClassicalFP<2u>&, ExactSummation_t<Spins>&);
template Array<complex_t> psi_O_k(const PsiClassicalFP<2u>&, const Spins&);
#endif
#if defined(ENABLE_PSI_CLASSICAL_ANN) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template std::complex<double> log_psi_s(const PsiClassicalANN<1u>&, const Spins&);
template Array<complex_t> psi_O_k_vector(const PsiClassicalANN<1u>&, ExactSummation_t<Spins>&);
template Array<complex_t> psi_O_k(const PsiClassicalANN<1u>&, const Spins&);
#endif
#if defined(ENABLE_PSI_CLASSICAL_ANN) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template std::complex<double> log_psi_s(const PsiClassicalANN<2u>&, const Spins&);
template Array<complex_t> psi_O_k_vector(const PsiClassicalANN<2u>&, ExactSummation_t<Spins>&);
template Array<complex_t> psi_O_k(const PsiClassicalANN<2u>&, const Spins&);
#endif
#if defined(ENABLE_PAULIS)
template std::complex<double> log_psi_s(const PsiDeep&, const PauliString&);
template Array<complex_t> psi_O_k_vector(const PsiDeep&, ExactSummation_t<PauliString>&);
template Array<complex_t> psi_O_k(const PsiDeep&, const PauliString&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template std::complex<double> log_psi_s(const PsiClassicalFP<1u>&, const PauliString&);
template Array<complex_t> psi_O_k_vector(const PsiClassicalFP<1u>&, ExactSummation_t<PauliString>&);
template Array<complex_t> psi_O_k(const PsiClassicalFP<1u>&, const PauliString&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template std::complex<double> log_psi_s(const PsiClassicalFP<2u>&, const PauliString&);
template Array<complex_t> psi_O_k_vector(const PsiClassicalFP<2u>&, ExactSummation_t<PauliString>&);
template Array<complex_t> psi_O_k(const PsiClassicalFP<2u>&, const PauliString&);
#endif
#if defined(ENABLE_PSI_CLASSICAL_ANN) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template std::complex<double> log_psi_s(const PsiClassicalANN<1u>&, const PauliString&);
template Array<complex_t> psi_O_k_vector(const PsiClassicalANN<1u>&, ExactSummation_t<PauliString>&);
template Array<complex_t> psi_O_k(const PsiClassicalANN<1u>&, const PauliString&);
#endif
#if defined(ENABLE_PSI_CLASSICAL_ANN) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template std::complex<double> log_psi_s(const PsiClassicalANN<2u>&, const PauliString&);
template Array<complex_t> psi_O_k_vector(const PsiClassicalANN<2u>&, ExactSummation_t<PauliString>&);
template Array<complex_t> psi_O_k(const PsiClassicalANN<2u>&, const PauliString&);
#endif

} // namespace ann_on_gpu
