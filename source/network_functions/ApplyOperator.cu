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
Array<complex_t> apply_operator(Psi_t& psi, const Operator& op, Ensemble& ensemble) {
    Array<complex_t> result(ensemble.get_num_steps(), psi.gpu);
    result.clear();

    auto psi_kernel = psi.kernel();
    auto op_kernel = op.kernel();
    auto result_ptr = result.data();

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
                result_ptr[index] = exp(log_psi) * local_energy;
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


#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_DEEP)
template Array<complex_t> apply_operator(PsiDeep&, const Operator&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_RBM)
template Array<complex_t> apply_operator(PsiRBM&, const Operator&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CNN)
template Array<complex_t> apply_operator(PsiCNN&, const Operator&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiFullyPolarized&, const Operator&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiClassicalFP<1u>&, const Operator&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiClassicalFP<2u>&, const Operator&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template Array<complex_t> apply_operator(PsiClassicalANN<1u>&, const Operator&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template Array<complex_t> apply_operator(PsiClassicalANN<2u>&, const Operator&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_DEEP)
template Array<complex_t> apply_operator(PsiDeep&, const Operator&, MonteCarlo_tt<Fermions>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_RBM)
template Array<complex_t> apply_operator(PsiRBM&, const Operator&, MonteCarlo_tt<Fermions>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CNN)
template Array<complex_t> apply_operator(PsiCNN&, const Operator&, MonteCarlo_tt<Fermions>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiFullyPolarized&, const Operator&, MonteCarlo_tt<Fermions>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiClassicalFP<1u>&, const Operator&, MonteCarlo_tt<Fermions>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiClassicalFP<2u>&, const Operator&, MonteCarlo_tt<Fermions>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template Array<complex_t> apply_operator(PsiClassicalANN<1u>&, const Operator&, MonteCarlo_tt<Fermions>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template Array<complex_t> apply_operator(PsiClassicalANN<2u>&, const Operator&, MonteCarlo_tt<Fermions>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_DEEP)
template Array<complex_t> apply_operator(PsiDeep&, const Operator&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_RBM)
template Array<complex_t> apply_operator(PsiRBM&, const Operator&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CNN)
template Array<complex_t> apply_operator(PsiCNN&, const Operator&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiFullyPolarized&, const Operator&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiClassicalFP<1u>&, const Operator&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiClassicalFP<2u>&, const Operator&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template Array<complex_t> apply_operator(PsiClassicalANN<1u>&, const Operator&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template Array<complex_t> apply_operator(PsiClassicalANN<2u>&, const Operator&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_DEEP)
template Array<complex_t> apply_operator(PsiDeep&, const Operator&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_RBM)
template Array<complex_t> apply_operator(PsiRBM&, const Operator&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CNN)
template Array<complex_t> apply_operator(PsiCNN&, const Operator&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiFullyPolarized&, const Operator&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiClassicalFP<1u>&, const Operator&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiClassicalFP<2u>&, const Operator&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template Array<complex_t> apply_operator(PsiClassicalANN<1u>&, const Operator&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template Array<complex_t> apply_operator(PsiClassicalANN<2u>&, const Operator&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_DEEP)
template Array<complex_t> apply_operator(PsiDeep&, const Operator&, ExactSummation_t<Fermions>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_RBM)
template Array<complex_t> apply_operator(PsiRBM&, const Operator&, ExactSummation_t<Fermions>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CNN)
template Array<complex_t> apply_operator(PsiCNN&, const Operator&, ExactSummation_t<Fermions>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiFullyPolarized&, const Operator&, ExactSummation_t<Fermions>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiClassicalFP<1u>&, const Operator&, ExactSummation_t<Fermions>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiClassicalFP<2u>&, const Operator&, ExactSummation_t<Fermions>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template Array<complex_t> apply_operator(PsiClassicalANN<1u>&, const Operator&, ExactSummation_t<Fermions>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template Array<complex_t> apply_operator(PsiClassicalANN<2u>&, const Operator&, ExactSummation_t<Fermions>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_DEEP)
template Array<complex_t> apply_operator(PsiDeep&, const Operator&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_RBM)
template Array<complex_t> apply_operator(PsiRBM&, const Operator&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CNN)
template Array<complex_t> apply_operator(PsiCNN&, const Operator&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiFullyPolarized&, const Operator&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiClassicalFP<1u>&, const Operator&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template Array<complex_t> apply_operator(PsiClassicalFP<2u>&, const Operator&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template Array<complex_t> apply_operator(PsiClassicalANN<1u>&, const Operator&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template Array<complex_t> apply_operator(PsiClassicalANN<2u>&, const Operator&, ExactSummation_t<PauliString>&);
#endif


} // namespace ann_on_gpu
