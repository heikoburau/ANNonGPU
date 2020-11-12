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

    auto num_params = this->F_vector.size();
    auto op_kernel = op.kernel();
    auto psi_kernel = psi.kernel();
    auto E_local_ptr = this->E_local_ar.data();
    auto O_k_ptr = this->O_k_ar.data();
    auto S_ptr = this->S_matrix.data();
    auto F_ptr = this->F_vector.data();

    ensemble.foreach(
        psi,
        [=] __device__ __host__ (
            const unsigned int index,
            const typename Ensemble::Basis_t& configuration,
            const typename Psi_t::dtype log_psi,
            typename Psi_t::Payload& payload,
            const typename Psi_t::real_dtype weight
        ) {
            #include "cuda_kernel_defines.h"

            SHARED complex_t local_energy;
            op_kernel.local_energy(local_energy, psi_kernel, configuration, log_psi, payload);

            SINGLE {
                generic_atomicAdd(E_local_ptr, weight * local_energy);
            }

            psi_kernel.init_payload(payload, configuration);
            psi_kernel.foreach_O_k(
                configuration,
                payload,
                [&](const unsigned int k, const complex_t& O_k) {
                    generic_atomicAdd(&O_k_ptr[k], weight * O_k);
                    generic_atomicAdd(&F_ptr[k], weight * local_energy * conj(O_k));

                    for(auto k_prime = 0u; k_prime < psi_kernel.num_params; k_prime++) {
                        generic_atomicAdd(
                            &S_ptr[k * num_params + k_prime],
                            weight * conj(O_k) * psi_kernel.get_O_k(k_prime, payload)
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
