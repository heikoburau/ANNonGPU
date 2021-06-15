// ***********************************************************
// *       This is an automatically generated file.          *
// *       For editing, please use the source file:          *
// PsiVector.cu.template
// ***********************************************************

#include "network_functions/PsiVector.hpp"
#include "quantum_states.hpp"
#include "ensembles.hpp"
#include "types.h"

#include <iostream>

namespace ann_on_gpu {


using namespace cuda_complex;


template<typename Psi_t, typename Ensemble>
Array<complex_t> log_psi_vector(Psi_t& psi, Ensemble& ensemble) {

    Array<complex_t> result(ensemble.get_num_steps(), ensemble.gpu);

    auto result_ptr = result.data();
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
                result_ptr[conf_index] = log_psi;
            }
        }
    );
    result.update_host();

    return result;
}


template<typename Psi_t, typename Ensemble>
Array<complex_t> psi_vector(Psi_t& psi, Ensemble& ensemble) {

    Array<complex_t> result(ensemble.get_num_steps(), ensemble.gpu);

    auto result_ptr = result.data();
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
                result_ptr[conf_index] = exp(log_psi);
            }
        }
    );
    result.update_host();

    return result;
}

template<typename Psi_t, typename Ensemble>
typename std_dtype<typename Psi_t::dtype>::type log_psi(Psi_t& psi, Ensemble& ensemble) {
    Array<typename Psi_t::dtype> result(1u, ensemble.gpu);

    result.clear();
    auto result_ptr = result.data();

    ensemble.foreach(
        psi,
        [=] __host__ __device__ (
            const unsigned int conf_index,
            const typename Ensemble::Basis_t& configuration,
            const typename Psi_t::dtype& log_psi,
            typename Psi_t::Payload& payload,
            const double weight
        ) {
            #include "cuda_kernel_defines.h"

            SINGLE {
                generic_atomicAdd(result_ptr, weight * log_psi);
            }
        }
    );
    result.update_host();

    return to_std(result.front());
}

template<typename Psi_t, typename Basis_t>
typename std_dtype<typename Psi_t::dtype>::type log_psi_s(Psi_t& psi, const Basis_t& configuration) {
    Array<typename Psi_t::dtype> result(1u, psi.gpu);

    auto result_ptr = result.data();
    auto psi_kernel = psi.kernel();
    auto conf = configuration;

    const auto functor = [=] __host__ __device__ () {
        #include "cuda_kernel_defines.h"

        SHARED typename Psi_t::Payload payload;
        SHARED typename Psi_t::dtype   log_psi;

        psi_kernel.init_payload(payload, conf, 0u);
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

    return to_std(result.front());
}


#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_DEEP)
template typename std_dtype<typename PsiDeep::dtype>::type log_psi_s(PsiDeep&, const Spins&);
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_RBM)
template typename std_dtype<typename PsiRBM::dtype>::type log_psi_s(PsiRBM&, const Spins&);
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CNN)
template typename std_dtype<typename PsiCNN::dtype>::type log_psi_s(PsiCNN&, const Spins&);
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template typename std_dtype<typename PsiFullyPolarized::dtype>::type log_psi_s(PsiFullyPolarized&, const Spins&);
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template typename std_dtype<typename PsiClassicalFP<1u>::dtype>::type log_psi_s(PsiClassicalFP<1u>&, const Spins&);
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template typename std_dtype<typename PsiClassicalFP<2u>::dtype>::type log_psi_s(PsiClassicalFP<2u>&, const Spins&);
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template typename std_dtype<typename PsiClassicalANN<1u>::dtype>::type log_psi_s(PsiClassicalANN<1u>&, const Spins&);
#endif
#if defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template typename std_dtype<typename PsiClassicalANN<2u>::dtype>::type log_psi_s(PsiClassicalANN<2u>&, const Spins&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_DEEP)
template typename std_dtype<typename PsiDeep::dtype>::type log_psi_s(PsiDeep&, const PauliString&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_RBM)
template typename std_dtype<typename PsiRBM::dtype>::type log_psi_s(PsiRBM&, const PauliString&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CNN)
template typename std_dtype<typename PsiCNN::dtype>::type log_psi_s(PsiCNN&, const PauliString&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template typename std_dtype<typename PsiFullyPolarized::dtype>::type log_psi_s(PsiFullyPolarized&, const PauliString&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template typename std_dtype<typename PsiClassicalFP<1u>::dtype>::type log_psi_s(PsiClassicalFP<1u>&, const PauliString&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template typename std_dtype<typename PsiClassicalFP<2u>::dtype>::type log_psi_s(PsiClassicalFP<2u>&, const PauliString&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template typename std_dtype<typename PsiClassicalANN<1u>::dtype>::type log_psi_s(PsiClassicalANN<1u>&, const PauliString&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template typename std_dtype<typename PsiClassicalANN<2u>::dtype>::type log_psi_s(PsiClassicalANN<2u>&, const PauliString&);
#endif

#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_DEEP)
template typename std_dtype<typename PsiDeep::dtype>::type log_psi(PsiDeep& psi, MonteCarlo_tt<Spins>& ensemble);
template Array<complex_t> log_psi_vector(PsiDeep& psi, MonteCarlo_tt<Spins>& ensemble);
template Array<complex_t> psi_vector(PsiDeep& psi, MonteCarlo_tt<Spins>& ensemble);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_RBM)
template typename std_dtype<typename PsiRBM::dtype>::type log_psi(PsiRBM& psi, MonteCarlo_tt<Spins>& ensemble);
template Array<complex_t> log_psi_vector(PsiRBM& psi, MonteCarlo_tt<Spins>& ensemble);
template Array<complex_t> psi_vector(PsiRBM& psi, MonteCarlo_tt<Spins>& ensemble);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CNN)
template typename std_dtype<typename PsiCNN::dtype>::type log_psi(PsiCNN& psi, MonteCarlo_tt<Spins>& ensemble);
template Array<complex_t> log_psi_vector(PsiCNN& psi, MonteCarlo_tt<Spins>& ensemble);
template Array<complex_t> psi_vector(PsiCNN& psi, MonteCarlo_tt<Spins>& ensemble);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template typename std_dtype<typename PsiFullyPolarized::dtype>::type log_psi(PsiFullyPolarized& psi, MonteCarlo_tt<Spins>& ensemble);
template Array<complex_t> log_psi_vector(PsiFullyPolarized& psi, MonteCarlo_tt<Spins>& ensemble);
template Array<complex_t> psi_vector(PsiFullyPolarized& psi, MonteCarlo_tt<Spins>& ensemble);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template typename std_dtype<typename PsiClassicalFP<1u>::dtype>::type log_psi(PsiClassicalFP<1u>& psi, MonteCarlo_tt<Spins>& ensemble);
template Array<complex_t> log_psi_vector(PsiClassicalFP<1u>& psi, MonteCarlo_tt<Spins>& ensemble);
template Array<complex_t> psi_vector(PsiClassicalFP<1u>& psi, MonteCarlo_tt<Spins>& ensemble);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template typename std_dtype<typename PsiClassicalFP<2u>::dtype>::type log_psi(PsiClassicalFP<2u>& psi, MonteCarlo_tt<Spins>& ensemble);
template Array<complex_t> log_psi_vector(PsiClassicalFP<2u>& psi, MonteCarlo_tt<Spins>& ensemble);
template Array<complex_t> psi_vector(PsiClassicalFP<2u>& psi, MonteCarlo_tt<Spins>& ensemble);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template typename std_dtype<typename PsiClassicalANN<1u>::dtype>::type log_psi(PsiClassicalANN<1u>& psi, MonteCarlo_tt<Spins>& ensemble);
template Array<complex_t> log_psi_vector(PsiClassicalANN<1u>& psi, MonteCarlo_tt<Spins>& ensemble);
template Array<complex_t> psi_vector(PsiClassicalANN<1u>& psi, MonteCarlo_tt<Spins>& ensemble);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template typename std_dtype<typename PsiClassicalANN<2u>::dtype>::type log_psi(PsiClassicalANN<2u>& psi, MonteCarlo_tt<Spins>& ensemble);
template Array<complex_t> log_psi_vector(PsiClassicalANN<2u>& psi, MonteCarlo_tt<Spins>& ensemble);
template Array<complex_t> psi_vector(PsiClassicalANN<2u>& psi, MonteCarlo_tt<Spins>& ensemble);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_DEEP)
template typename std_dtype<typename PsiDeep::dtype>::type log_psi(PsiDeep& psi, MonteCarlo_tt<PauliString>& ensemble);
template Array<complex_t> log_psi_vector(PsiDeep& psi, MonteCarlo_tt<PauliString>& ensemble);
template Array<complex_t> psi_vector(PsiDeep& psi, MonteCarlo_tt<PauliString>& ensemble);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_RBM)
template typename std_dtype<typename PsiRBM::dtype>::type log_psi(PsiRBM& psi, MonteCarlo_tt<PauliString>& ensemble);
template Array<complex_t> log_psi_vector(PsiRBM& psi, MonteCarlo_tt<PauliString>& ensemble);
template Array<complex_t> psi_vector(PsiRBM& psi, MonteCarlo_tt<PauliString>& ensemble);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CNN)
template typename std_dtype<typename PsiCNN::dtype>::type log_psi(PsiCNN& psi, MonteCarlo_tt<PauliString>& ensemble);
template Array<complex_t> log_psi_vector(PsiCNN& psi, MonteCarlo_tt<PauliString>& ensemble);
template Array<complex_t> psi_vector(PsiCNN& psi, MonteCarlo_tt<PauliString>& ensemble);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template typename std_dtype<typename PsiFullyPolarized::dtype>::type log_psi(PsiFullyPolarized& psi, MonteCarlo_tt<PauliString>& ensemble);
template Array<complex_t> log_psi_vector(PsiFullyPolarized& psi, MonteCarlo_tt<PauliString>& ensemble);
template Array<complex_t> psi_vector(PsiFullyPolarized& psi, MonteCarlo_tt<PauliString>& ensemble);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template typename std_dtype<typename PsiClassicalFP<1u>::dtype>::type log_psi(PsiClassicalFP<1u>& psi, MonteCarlo_tt<PauliString>& ensemble);
template Array<complex_t> log_psi_vector(PsiClassicalFP<1u>& psi, MonteCarlo_tt<PauliString>& ensemble);
template Array<complex_t> psi_vector(PsiClassicalFP<1u>& psi, MonteCarlo_tt<PauliString>& ensemble);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template typename std_dtype<typename PsiClassicalFP<2u>::dtype>::type log_psi(PsiClassicalFP<2u>& psi, MonteCarlo_tt<PauliString>& ensemble);
template Array<complex_t> log_psi_vector(PsiClassicalFP<2u>& psi, MonteCarlo_tt<PauliString>& ensemble);
template Array<complex_t> psi_vector(PsiClassicalFP<2u>& psi, MonteCarlo_tt<PauliString>& ensemble);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template typename std_dtype<typename PsiClassicalANN<1u>::dtype>::type log_psi(PsiClassicalANN<1u>& psi, MonteCarlo_tt<PauliString>& ensemble);
template Array<complex_t> log_psi_vector(PsiClassicalANN<1u>& psi, MonteCarlo_tt<PauliString>& ensemble);
template Array<complex_t> psi_vector(PsiClassicalANN<1u>& psi, MonteCarlo_tt<PauliString>& ensemble);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template typename std_dtype<typename PsiClassicalANN<2u>::dtype>::type log_psi(PsiClassicalANN<2u>& psi, MonteCarlo_tt<PauliString>& ensemble);
template Array<complex_t> log_psi_vector(PsiClassicalANN<2u>& psi, MonteCarlo_tt<PauliString>& ensemble);
template Array<complex_t> psi_vector(PsiClassicalANN<2u>& psi, MonteCarlo_tt<PauliString>& ensemble);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_DEEP)
template typename std_dtype<typename PsiDeep::dtype>::type log_psi(PsiDeep& psi, ExactSummation_t<Spins>& ensemble);
template Array<complex_t> log_psi_vector(PsiDeep& psi, ExactSummation_t<Spins>& ensemble);
template Array<complex_t> psi_vector(PsiDeep& psi, ExactSummation_t<Spins>& ensemble);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_RBM)
template typename std_dtype<typename PsiRBM::dtype>::type log_psi(PsiRBM& psi, ExactSummation_t<Spins>& ensemble);
template Array<complex_t> log_psi_vector(PsiRBM& psi, ExactSummation_t<Spins>& ensemble);
template Array<complex_t> psi_vector(PsiRBM& psi, ExactSummation_t<Spins>& ensemble);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CNN)
template typename std_dtype<typename PsiCNN::dtype>::type log_psi(PsiCNN& psi, ExactSummation_t<Spins>& ensemble);
template Array<complex_t> log_psi_vector(PsiCNN& psi, ExactSummation_t<Spins>& ensemble);
template Array<complex_t> psi_vector(PsiCNN& psi, ExactSummation_t<Spins>& ensemble);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template typename std_dtype<typename PsiFullyPolarized::dtype>::type log_psi(PsiFullyPolarized& psi, ExactSummation_t<Spins>& ensemble);
template Array<complex_t> log_psi_vector(PsiFullyPolarized& psi, ExactSummation_t<Spins>& ensemble);
template Array<complex_t> psi_vector(PsiFullyPolarized& psi, ExactSummation_t<Spins>& ensemble);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template typename std_dtype<typename PsiClassicalFP<1u>::dtype>::type log_psi(PsiClassicalFP<1u>& psi, ExactSummation_t<Spins>& ensemble);
template Array<complex_t> log_psi_vector(PsiClassicalFP<1u>& psi, ExactSummation_t<Spins>& ensemble);
template Array<complex_t> psi_vector(PsiClassicalFP<1u>& psi, ExactSummation_t<Spins>& ensemble);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template typename std_dtype<typename PsiClassicalFP<2u>::dtype>::type log_psi(PsiClassicalFP<2u>& psi, ExactSummation_t<Spins>& ensemble);
template Array<complex_t> log_psi_vector(PsiClassicalFP<2u>& psi, ExactSummation_t<Spins>& ensemble);
template Array<complex_t> psi_vector(PsiClassicalFP<2u>& psi, ExactSummation_t<Spins>& ensemble);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template typename std_dtype<typename PsiClassicalANN<1u>::dtype>::type log_psi(PsiClassicalANN<1u>& psi, ExactSummation_t<Spins>& ensemble);
template Array<complex_t> log_psi_vector(PsiClassicalANN<1u>& psi, ExactSummation_t<Spins>& ensemble);
template Array<complex_t> psi_vector(PsiClassicalANN<1u>& psi, ExactSummation_t<Spins>& ensemble);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template typename std_dtype<typename PsiClassicalANN<2u>::dtype>::type log_psi(PsiClassicalANN<2u>& psi, ExactSummation_t<Spins>& ensemble);
template Array<complex_t> log_psi_vector(PsiClassicalANN<2u>& psi, ExactSummation_t<Spins>& ensemble);
template Array<complex_t> psi_vector(PsiClassicalANN<2u>& psi, ExactSummation_t<Spins>& ensemble);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_DEEP)
template typename std_dtype<typename PsiDeep::dtype>::type log_psi(PsiDeep& psi, ExactSummation_t<PauliString>& ensemble);
template Array<complex_t> log_psi_vector(PsiDeep& psi, ExactSummation_t<PauliString>& ensemble);
template Array<complex_t> psi_vector(PsiDeep& psi, ExactSummation_t<PauliString>& ensemble);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_RBM)
template typename std_dtype<typename PsiRBM::dtype>::type log_psi(PsiRBM& psi, ExactSummation_t<PauliString>& ensemble);
template Array<complex_t> log_psi_vector(PsiRBM& psi, ExactSummation_t<PauliString>& ensemble);
template Array<complex_t> psi_vector(PsiRBM& psi, ExactSummation_t<PauliString>& ensemble);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CNN)
template typename std_dtype<typename PsiCNN::dtype>::type log_psi(PsiCNN& psi, ExactSummation_t<PauliString>& ensemble);
template Array<complex_t> log_psi_vector(PsiCNN& psi, ExactSummation_t<PauliString>& ensemble);
template Array<complex_t> psi_vector(PsiCNN& psi, ExactSummation_t<PauliString>& ensemble);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template typename std_dtype<typename PsiFullyPolarized::dtype>::type log_psi(PsiFullyPolarized& psi, ExactSummation_t<PauliString>& ensemble);
template Array<complex_t> log_psi_vector(PsiFullyPolarized& psi, ExactSummation_t<PauliString>& ensemble);
template Array<complex_t> psi_vector(PsiFullyPolarized& psi, ExactSummation_t<PauliString>& ensemble);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template typename std_dtype<typename PsiClassicalFP<1u>::dtype>::type log_psi(PsiClassicalFP<1u>& psi, ExactSummation_t<PauliString>& ensemble);
template Array<complex_t> log_psi_vector(PsiClassicalFP<1u>& psi, ExactSummation_t<PauliString>& ensemble);
template Array<complex_t> psi_vector(PsiClassicalFP<1u>& psi, ExactSummation_t<PauliString>& ensemble);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template typename std_dtype<typename PsiClassicalFP<2u>::dtype>::type log_psi(PsiClassicalFP<2u>& psi, ExactSummation_t<PauliString>& ensemble);
template Array<complex_t> log_psi_vector(PsiClassicalFP<2u>& psi, ExactSummation_t<PauliString>& ensemble);
template Array<complex_t> psi_vector(PsiClassicalFP<2u>& psi, ExactSummation_t<PauliString>& ensemble);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template typename std_dtype<typename PsiClassicalANN<1u>::dtype>::type log_psi(PsiClassicalANN<1u>& psi, ExactSummation_t<PauliString>& ensemble);
template Array<complex_t> log_psi_vector(PsiClassicalANN<1u>& psi, ExactSummation_t<PauliString>& ensemble);
template Array<complex_t> psi_vector(PsiClassicalANN<1u>& psi, ExactSummation_t<PauliString>& ensemble);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template typename std_dtype<typename PsiClassicalANN<2u>::dtype>::type log_psi(PsiClassicalANN<2u>& psi, ExactSummation_t<PauliString>& ensemble);
template Array<complex_t> log_psi_vector(PsiClassicalANN<2u>& psi, ExactSummation_t<PauliString>& ensemble);
template Array<complex_t> psi_vector(PsiClassicalANN<2u>& psi, ExactSummation_t<PauliString>& ensemble);
#endif


} // namespace ann_on_gpu
