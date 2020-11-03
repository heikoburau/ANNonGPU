// ***********************************************************
// *       This is an automatically generated file.          *
// *       For editing, please use the source file:          *
// ExpectationValue.cu.template
// ***********************************************************

#include "network_functions/ExpectationValue.hpp"
#include "ensembles.hpp"
#include "quantum_states.hpp"
#include "Array.hpp"


namespace ann_on_gpu {

ExpectationValue::ExpectationValue(const bool gpu) : A_local(1, gpu), A_local_abs2(1, gpu) {
}

template<typename Psi_t, typename Ensemble>
complex<double> ExpectationValue::operator()(
    const Operator& operator_, const Psi_t& psi, Ensemble& ensemble
) {

    this->A_local.clear();
    auto A_local_ptr = this->A_local.data();
    auto psi_kernel = psi.kernel();
    auto op_kernel = operator_.kernel();

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

            SHARED typename Psi_t::dtype local_energy;
            op_kernel.local_energy(local_energy, psi_kernel, configuration, log_psi, payload);

            SINGLE {
                generic_atomicAdd(A_local_ptr, weight * local_energy);
            }
        }
    );

    this->A_local.update_host();
    return this->A_local.front().to_std();
}


template<typename Psi_t, typename Ensemble>
pair<double, complex<double>> ExpectationValue::fluctuation(
    const Operator& operator_, const Psi_t& psi, Ensemble& ensemble
) {

    this->A_local.clear();
    this->A_local_abs2.clear();

    auto A_local_ptr = this->A_local.data();
    auto A_local_abs2_ptr = this->A_local_abs2.data();
    auto psi_kernel = psi.kernel();
    auto op_kernel = operator_.kernel();

    ensemble.foreach(
        psi,
        [=] __device__ __host__ (
            const unsigned int spin_index,
            const typename Ensemble::Basis_t& configuration,
            const typename Psi_t::dtype log_psi,
            typename Psi_t::Payload& payload,
            const typename Psi_t::real_dtype weight
        ) {
            #include "cuda_kernel_defines.h"

            SHARED typename Psi_t::dtype local_energy;
            op_kernel.local_energy(local_energy, psi_kernel, configuration, log_psi, payload);

            SINGLE {
                generic_atomicAdd(A_local_ptr, weight * local_energy);
                generic_atomicAdd(A_local_abs2_ptr, weight * abs2(local_energy));
            }
        }
    );

    this->A_local.update_host();
    this->A_local_abs2.update_host();

    return {
        sqrt(this->A_local_abs2.front() - abs2(this->A_local.front())),
        this->A_local.front().to_std()
    };
}

#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS)
template complex<double> ExpectationValue::operator()(const Operator&, const PsiDeep& psi, MonteCarlo_tt<Spins>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator&, const PsiDeep&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS)
template complex<double> ExpectationValue::operator()(const Operator&, const PsiFullyPolarized& psi, MonteCarlo_tt<Spins>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator&, const PsiFullyPolarized&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS)
template complex<double> ExpectationValue::operator()(const Operator&, const PsiClassicalFP<1u>& psi, MonteCarlo_tt<Spins>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator&, const PsiClassicalFP<1u>&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS)
template complex<double> ExpectationValue::operator()(const Operator&, const PsiClassicalFP<2u>& psi, MonteCarlo_tt<Spins>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator&, const PsiClassicalFP<2u>&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_PSI_CLASSICAL_ANN) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS)
template complex<double> ExpectationValue::operator()(const Operator&, const PsiClassicalANN<1u>& psi, MonteCarlo_tt<Spins>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator&, const PsiClassicalANN<1u>&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_PSI_CLASSICAL_ANN) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS)
template complex<double> ExpectationValue::operator()(const Operator&, const PsiClassicalANN<2u>& psi, MonteCarlo_tt<Spins>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator&, const PsiClassicalANN<2u>&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_MONTE_CARLO)
template complex<double> ExpectationValue::operator()(const Operator&, const PsiDeep& psi, MonteCarlo_tt<PauliString>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator&, const PsiDeep&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_MONTE_CARLO)
template complex<double> ExpectationValue::operator()(const Operator&, const PsiFullyPolarized& psi, MonteCarlo_tt<PauliString>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator&, const PsiFullyPolarized&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_MONTE_CARLO)
template complex<double> ExpectationValue::operator()(const Operator&, const PsiClassicalFP<1u>& psi, MonteCarlo_tt<PauliString>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator&, const PsiClassicalFP<1u>&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_MONTE_CARLO)
template complex<double> ExpectationValue::operator()(const Operator&, const PsiClassicalFP<2u>& psi, MonteCarlo_tt<PauliString>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator&, const PsiClassicalFP<2u>&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL_ANN) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_MONTE_CARLO)
template complex<double> ExpectationValue::operator()(const Operator&, const PsiClassicalANN<1u>& psi, MonteCarlo_tt<PauliString>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator&, const PsiClassicalANN<1u>&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL_ANN) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_MONTE_CARLO)
template complex<double> ExpectationValue::operator()(const Operator&, const PsiClassicalANN<2u>& psi, MonteCarlo_tt<PauliString>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator&, const PsiClassicalANN<2u>&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS)
template complex<double> ExpectationValue::operator()(const Operator&, const PsiDeep& psi, ExactSummation_t<Spins>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator&, const PsiDeep&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_SPINS)
template complex<double> ExpectationValue::operator()(const Operator&, const PsiFullyPolarized& psi, ExactSummation_t<Spins>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator&, const PsiFullyPolarized&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_SPINS)
template complex<double> ExpectationValue::operator()(const Operator&, const PsiClassicalFP<1u>& psi, ExactSummation_t<Spins>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator&, const PsiClassicalFP<1u>&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_SPINS)
template complex<double> ExpectationValue::operator()(const Operator&, const PsiClassicalFP<2u>& psi, ExactSummation_t<Spins>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator&, const PsiClassicalFP<2u>&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PSI_CLASSICAL_ANN) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_SPINS)
template complex<double> ExpectationValue::operator()(const Operator&, const PsiClassicalANN<1u>& psi, ExactSummation_t<Spins>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator&, const PsiClassicalANN<1u>&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PSI_CLASSICAL_ANN) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_SPINS)
template complex<double> ExpectationValue::operator()(const Operator&, const PsiClassicalANN<2u>& psi, ExactSummation_t<Spins>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator&, const PsiClassicalANN<2u>&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS)
template complex<double> ExpectationValue::operator()(const Operator&, const PsiDeep& psi, ExactSummation_t<PauliString>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator&, const PsiDeep&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template complex<double> ExpectationValue::operator()(const Operator&, const PsiFullyPolarized& psi, ExactSummation_t<PauliString>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator&, const PsiFullyPolarized&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template complex<double> ExpectationValue::operator()(const Operator&, const PsiClassicalFP<1u>& psi, ExactSummation_t<PauliString>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator&, const PsiClassicalFP<1u>&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template complex<double> ExpectationValue::operator()(const Operator&, const PsiClassicalFP<2u>& psi, ExactSummation_t<PauliString>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator&, const PsiClassicalFP<2u>&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PSI_CLASSICAL_ANN) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template complex<double> ExpectationValue::operator()(const Operator&, const PsiClassicalANN<1u>& psi, ExactSummation_t<PauliString>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator&, const PsiClassicalANN<1u>&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PSI_CLASSICAL_ANN) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template complex<double> ExpectationValue::operator()(const Operator&, const PsiClassicalANN<2u>& psi, ExactSummation_t<PauliString>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator&, const PsiClassicalANN<2u>&, ExactSummation_t<PauliString>&);
#endif

} // namespace ann_on_gpu
