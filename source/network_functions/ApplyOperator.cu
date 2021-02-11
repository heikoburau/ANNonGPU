// ***********************************************************
// *       This is an automatically generated file.          *
// *       For editing, please use the source file:          *
// ApplyOperator.cu.template
// ***********************************************************

#include "network_functions/ApplyOperator.hpp"
#include "quantum_states.hpp"
#include "ensembles.hpp"


namespace ann_on_gpu {


template<typename Psi_t, typename Ensemble>
Array<complex_t> apply_operator(Psi_t& psi, const Operator_t& op, Ensemble& ensemble) {
    Array<complex_t> result(ensemble.get_num_steps(), psi.gpu);
    result.clear();

    auto psi_kernel = psi.kernel();
    auto op_kernel = op.kernel();
    auto result_ptr = result.data();
    auto prefactor = psi.prefactor;

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
                result_ptr[index] = prefactor * exp(log_psi) * local_energy;
                // generic_atomicAdd(
                //     &result_ptr[index],
                //     weight * local_energy
                // );
            }
        }
    );

    result.update_host();
    return result;
}


#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS)
template Array<complex_t> apply_operator(PsiDeep&, const Operator_t&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiFullyPolarized&, const Operator_t&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiClassicalFP<1u>&, const Operator_t&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiClassicalFP<2u>&, const Operator_t&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template Array<complex_t> apply_operator(PsiClassicalANN<1u>&, const Operator_t&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template Array<complex_t> apply_operator(PsiClassicalANN<2u>&, const Operator_t&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS)
template Array<complex_t> apply_operator(PsiDeep&, const Operator_t&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiFullyPolarized&, const Operator_t&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiClassicalFP<1u>&, const Operator_t&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiClassicalFP<2u>&, const Operator_t&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template Array<complex_t> apply_operator(PsiClassicalANN<1u>&, const Operator_t&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template Array<complex_t> apply_operator(PsiClassicalANN<2u>&, const Operator_t&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS)
template Array<complex_t> apply_operator(PsiDeep&, const Operator_t&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiFullyPolarized&, const Operator_t&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiClassicalFP<1u>&, const Operator_t&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiClassicalFP<2u>&, const Operator_t&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template Array<complex_t> apply_operator(PsiClassicalANN<1u>&, const Operator_t&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template Array<complex_t> apply_operator(PsiClassicalANN<2u>&, const Operator_t&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS)
template Array<complex_t> apply_operator(PsiDeep&, const Operator_t&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiFullyPolarized&, const Operator_t&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiClassicalFP<1u>&, const Operator_t&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiClassicalFP<2u>&, const Operator_t&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template Array<complex_t> apply_operator(PsiClassicalANN<1u>&, const Operator_t&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template Array<complex_t> apply_operator(PsiClassicalANN<2u>&, const Operator_t&, ExactSummation_t<PauliString>&);
#endif


} // namespace ann_on_gpu
