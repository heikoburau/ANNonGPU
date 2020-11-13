// ***********************************************************
// *       This is an automatically generated file.          *
// *       For editing, please use the source file:          *
// TDVP.cu.template
// ***********************************************************

#ifndef LEAN_AND_MEAN

#include "network_functions/TDVP.hpp"
#include "quantum_states.hpp"
#include "ensembles.hpp"


namespace ann_on_gpu {


template<typename Psi_t, typename Ensemble>
void TDVP::eval(const Operator& op, const Psi_t& psi, Ensemble& ensemble) {
    this->E_local_ar.clear();
    this->O_k_ar.clear();
    this->S_matrix.clear();
    this->F_vector.clear();
    this->prob_ratio.clear();

    auto num_params = this->F_vector.size();
    auto op_kernel = op.kernel();
    auto psi_kernel = psi.kernel();
    auto psi_ref_kernel = psi.psi_ref.kernel();
    auto E_local_ptr = this->E_local_ar.data();
    auto O_k_ptr = this->O_k_ar.data();
    auto S_ptr = this->S_matrix.data();
    auto F_ptr = this->F_vector.data();
    auto prob_ratio_ptr = this->prob_ratio.data();

    using PsiRef = typename Psi_t::PsiRef;

    ensemble.foreach(
        psi.psi_ref,
        [=] __device__ __host__ (
            const unsigned int index,
            const typename Ensemble::Basis_t& configuration,
            const typename PsiRef::dtype log_psi_ref,
            typename PsiRef::Payload& payload_ref,
            const typename PsiRef::real_dtype weight
        ) {
            #include "cuda_kernel_defines.h"

            SHARED complex_t                   log_psi;
            SHARED typename Psi_t::Payload     payload;

            SHARED complex_t local_energy;
            psi_kernel.log_psi_s(log_psi, configuration, payload);
            op_kernel.local_energy(local_energy, psi_kernel, configuration, log_psi, payload);

            SHARED double prob_ratio;

            SINGLE {
                prob_ratio = exp(2.0 * (log_psi.real() - log_psi_ref.real()));
                generic_atomicAdd(prob_ratio_ptr, weight * prob_ratio);
                generic_atomicAdd(E_local_ptr, weight * prob_ratio * local_energy);
            }

            psi_kernel.init_payload(payload, configuration);
            psi_kernel.foreach_O_k(
                configuration,
                payload,
                [&](const unsigned int k, const complex_t& O_k) {
                    generic_atomicAdd(&O_k_ptr[k], weight * prob_ratio * O_k);
                    generic_atomicAdd(&F_ptr[k], weight * prob_ratio * local_energy * conj(O_k));

                    for(auto k_prime = 0u; k_prime < psi_kernel.num_params; k_prime++) {
                        generic_atomicAdd(
                            &S_ptr[k * num_params + k_prime],
                            weight * prob_ratio * conj(O_k) * psi_kernel.get_O_k(k_prime, payload)
                        );
                    }
                }
            );
        }
    );

    this->E_local_ar.update_host();
    this->O_k_ar.update_host();
    this->S_matrix.update_host();
    this->F_vector.update_host();
    this->prob_ratio.update_host();

    this->E_local_ar.front() /= this->prob_ratio.front();
    for(auto k = 0u; k < num_params; k++) {
        this->O_k_ar[k] /= this->prob_ratio.front();
        this->F_vector[k] /= this->prob_ratio.front();

        for(auto k_prime = 0u; k_prime < num_params; k_prime++) {
            this->S_matrix[k * num_params + k_prime] /= this->prob_ratio.front();
        }
    }

    for(auto k = 0u; k < num_params; k++) {
        for(auto k_prime = 0u; k_prime < num_params; k_prime++) {
            this->S_matrix[k * num_params + k_prime] -= (
                conj(this->O_k_ar[k]) * this->O_k_ar[k_prime]
            );
        }

        this->F_vector[k] -= this->E_local_ar.front() * conj(this->O_k_ar[k]);
    }
}


#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, const PsiClassicalFP<1u>&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, const PsiClassicalFP<2u>&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, const PsiClassicalANN<1u>&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, const PsiClassicalANN<2u>&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, const PsiClassicalFP<1u>&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, const PsiClassicalFP<2u>&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, const PsiClassicalANN<1u>&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, const PsiClassicalANN<2u>&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, const PsiClassicalFP<1u>&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, const PsiClassicalFP<2u>&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, const PsiClassicalANN<1u>&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, const PsiClassicalANN<2u>&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, const PsiClassicalFP<1u>&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, const PsiClassicalFP<2u>&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, const PsiClassicalANN<1u>&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, const PsiClassicalANN<2u>&, ExactSummation_t<PauliString>&);
#endif


} // namespace ann_on_gpu


#endif // LEAN_AND_MEAN
