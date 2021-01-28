// ***********************************************************
// *       This is an automatically generated file.          *
// *       For editing, please use the source file:          *
// ExpectationValue.cu.template
// ***********************************************************

#ifndef LEAN_AND_MEAN

#include "network_functions/ExpectationValue.hpp"
#include "ensembles.hpp"
#include "quantum_states.hpp"
#include "Array.hpp"


namespace ann_on_gpu {

ExpectationValue::ExpectationValue(const bool gpu)
    :
    A_local(1, gpu),
    A_local_abs2(1, gpu),
    prob_ratio(1, gpu)
    {}


template<typename Psi_t, typename Ensemble>
complex<double> ExpectationValue::operator()(
    const Operator_t& operator_, Psi_t& psi, Ensemble& ensemble
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
Array<complex_t> ExpectationValue::operator()(
    const vector<Operator_t>& operator_array, Psi_t& psi, Ensemble& ensemble
) {
    Array<complex_t> result(operator_array.size(), psi.gpu);
    result.clear();
    auto result_ptr = result.data();

    auto psi_kernel = psi.kernel();

    Array<typename Operator_t::Kernel> operator_kernel_array(operator_array.size(), psi.gpu);
    for(auto i = 0u; i < operator_array.size(); i++) {
        operator_kernel_array[i] = operator_array[i].kernel();
    }
    operator_kernel_array.update_device();

    auto num_operators = operator_array.size();
    auto operator_kernel_ptr = operator_kernel_array.data();

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
            SHARED_MEM_LOOP_BEGIN(n, num_operators) {
                operator_kernel_ptr[n].local_energy(local_energy, psi_kernel, configuration, log_psi, payload);

                SINGLE {
                    generic_atomicAdd(&result_ptr[n], weight * local_energy);
                }

                SHARED_MEM_LOOP_END(n);
            }
        }
    );

    result.update_host();
    return result;
}

template<typename Psi_t, typename PsiSampling_t, typename Ensemble>
complex<double> ExpectationValue::operator()(
    const Operator_t& operator_, Psi_t& psi, PsiSampling_t& psi_sampling, Ensemble& ensemble
) {
    this->A_local.clear();
    this->prob_ratio.clear();

    auto A_local_ptr = this->A_local.data();
    auto psi_kernel = psi.kernel();
    auto op_kernel = operator_.kernel();

    ensemble.foreach(
        psi_sampling,
        [=] __device__ __host__ (
            const unsigned int index,
            const typename Ensemble::Basis_t& configuration,
            const typename PsiSampling_t::dtype& log_psi_sampling,
            typename PsiSampling_t::Payload& payload_sampling,
            const typename PsiSampling_t::real_dtype weight
        ) {
            #include "cuda_kernel_defines.h"

            SHARED typename Psi_t::dtype    local_energy;
            SHARED typename Psi_t::dtype    log_psi;
            SHARED typename Psi_t::Payload  payload;
            SHARED double                   prob_ratio;

            psi_kernel.init_payload(payload, configuration);
            psi_kernel.log_psi_s(log_psi, configuration, payload);
            op_kernel.local_energy(local_energy, psi_kernel, configuration, log_psi, payload);

            SINGLE {
                prob_ratio = exp(2.0 * (log_psi.real() - log_psi_sampling.real()));
                generic_atomicAdd(A_local_ptr, weight * prob_ratio * local_energy);
            }
        }
    );

    this->A_local.update_host();
    this->prob_ratio.update_host();

    return this->A_local.front().to_std() / this->prob_ratio.front();
}


template<typename Psi_t, typename Ensemble>
pair<double, complex<double>> ExpectationValue::fluctuation(
    const Operator_t& operator_, Psi_t& psi, Ensemble& ensemble
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


template<typename Psi_t, typename Ensemble>
Array<complex_t> ExpectationValue::gradient(
    const Operator_t& operator_, Psi_t& psi, Ensemble& ensemble
) {
    Array<complex_t> result(psi.num_params, psi.gpu);
    result.clear();

    auto result_ptr = result.data();
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

            psi_kernel.init_payload(payload, configuration);
            psi_kernel.foreach_O_k(
                configuration,
                payload,
                [&](const unsigned int k, const complex_t& O_k) {
                    generic_atomicAdd(&result_ptr[k], weight * 2.0 * conj(O_k) * local_energy);
                }
            );
        }
    );

    result.update_host();

    return result;
}


template<typename Psi_t, typename Ensemble>
pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(
    const Operator_t& operator_, Psi_t& psi, Ensemble& ensemble
) {
    Array<complex_t> result(psi.num_params, psi.gpu);
    Array<double> result2(psi.num_params, psi.gpu);

    result.clear();
    result2.clear();

    auto result_ptr = result.data();
    auto result2_ptr = result2.data();
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

            psi_kernel.init_payload(payload, configuration);
            psi_kernel.foreach_O_k(
                configuration,
                payload,
                [&](const unsigned int k, const complex_t& O_k) {
                    const auto val = 2.0 * conj(O_k) * local_energy;
                    generic_atomicAdd(&result_ptr[k], weight * val);
                    generic_atomicAdd(&result2_ptr[k], weight * abs2(val));
                }
            );
        }
    );

    result.update_host();
    result2.update_host();

    for(auto k = 0u; k < psi.num_params; k++) {
        result2[k] -= abs2(result[k]);
        result2[k] /= ensemble.get_num_steps();
        result2[k] = sqrt(result2[k]);
    }

    return make_pair(result, result2);
}

#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS)
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiDeep& psi, MonteCarlo_tt<Spins>&);
template Array<complex_t> ExpectationValue::operator()(const vector<Operator_t>&, PsiDeep& psi, MonteCarlo_tt<Spins>&);
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiDeep& psi, PsiDeep&, MonteCarlo_tt<Spins>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator_t&, PsiDeep&, MonteCarlo_tt<Spins>&);
template Array<complex_t> ExpectationValue::gradient(const Operator_t&, PsiDeep& psi, MonteCarlo_tt<Spins>&);
template pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(const Operator_t&, PsiDeep& psi, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiFullyPolarized& psi, MonteCarlo_tt<Spins>&);
template Array<complex_t> ExpectationValue::operator()(const vector<Operator_t>&, PsiFullyPolarized& psi, MonteCarlo_tt<Spins>&);
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiFullyPolarized& psi, PsiDeep&, MonteCarlo_tt<Spins>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator_t&, PsiFullyPolarized&, MonteCarlo_tt<Spins>&);
template Array<complex_t> ExpectationValue::gradient(const Operator_t&, PsiFullyPolarized& psi, MonteCarlo_tt<Spins>&);
template pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(const Operator_t&, PsiFullyPolarized& psi, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalFP<1u>& psi, MonteCarlo_tt<Spins>&);
template Array<complex_t> ExpectationValue::operator()(const vector<Operator_t>&, PsiClassicalFP<1u>& psi, MonteCarlo_tt<Spins>&);
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalFP<1u>& psi, PsiDeep&, MonteCarlo_tt<Spins>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator_t&, PsiClassicalFP<1u>&, MonteCarlo_tt<Spins>&);
template Array<complex_t> ExpectationValue::gradient(const Operator_t&, PsiClassicalFP<1u>& psi, MonteCarlo_tt<Spins>&);
template pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(const Operator_t&, PsiClassicalFP<1u>& psi, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalFP<2u>& psi, MonteCarlo_tt<Spins>&);
template Array<complex_t> ExpectationValue::operator()(const vector<Operator_t>&, PsiClassicalFP<2u>& psi, MonteCarlo_tt<Spins>&);
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalFP<2u>& psi, PsiDeep&, MonteCarlo_tt<Spins>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator_t&, PsiClassicalFP<2u>&, MonteCarlo_tt<Spins>&);
template Array<complex_t> ExpectationValue::gradient(const Operator_t&, PsiClassicalFP<2u>& psi, MonteCarlo_tt<Spins>&);
template pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(const Operator_t&, PsiClassicalFP<2u>& psi, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalANN<1u>& psi, MonteCarlo_tt<Spins>&);
template Array<complex_t> ExpectationValue::operator()(const vector<Operator_t>&, PsiClassicalANN<1u>& psi, MonteCarlo_tt<Spins>&);
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalANN<1u>& psi, PsiDeep&, MonteCarlo_tt<Spins>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator_t&, PsiClassicalANN<1u>&, MonteCarlo_tt<Spins>&);
template Array<complex_t> ExpectationValue::gradient(const Operator_t&, PsiClassicalANN<1u>& psi, MonteCarlo_tt<Spins>&);
template pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(const Operator_t&, PsiClassicalANN<1u>& psi, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalANN<2u>& psi, MonteCarlo_tt<Spins>&);
template Array<complex_t> ExpectationValue::operator()(const vector<Operator_t>&, PsiClassicalANN<2u>& psi, MonteCarlo_tt<Spins>&);
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalANN<2u>& psi, PsiDeep&, MonteCarlo_tt<Spins>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator_t&, PsiClassicalANN<2u>&, MonteCarlo_tt<Spins>&);
template Array<complex_t> ExpectationValue::gradient(const Operator_t&, PsiClassicalANN<2u>& psi, MonteCarlo_tt<Spins>&);
template pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(const Operator_t&, PsiClassicalANN<2u>& psi, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS)
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiDeep& psi, MonteCarlo_tt<PauliString>&);
template Array<complex_t> ExpectationValue::operator()(const vector<Operator_t>&, PsiDeep& psi, MonteCarlo_tt<PauliString>&);
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiDeep& psi, PsiDeep&, MonteCarlo_tt<PauliString>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator_t&, PsiDeep&, MonteCarlo_tt<PauliString>&);
template Array<complex_t> ExpectationValue::gradient(const Operator_t&, PsiDeep& psi, MonteCarlo_tt<PauliString>&);
template pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(const Operator_t&, PsiDeep& psi, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiFullyPolarized& psi, MonteCarlo_tt<PauliString>&);
template Array<complex_t> ExpectationValue::operator()(const vector<Operator_t>&, PsiFullyPolarized& psi, MonteCarlo_tt<PauliString>&);
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiFullyPolarized& psi, PsiDeep&, MonteCarlo_tt<PauliString>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator_t&, PsiFullyPolarized&, MonteCarlo_tt<PauliString>&);
template Array<complex_t> ExpectationValue::gradient(const Operator_t&, PsiFullyPolarized& psi, MonteCarlo_tt<PauliString>&);
template pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(const Operator_t&, PsiFullyPolarized& psi, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalFP<1u>& psi, MonteCarlo_tt<PauliString>&);
template Array<complex_t> ExpectationValue::operator()(const vector<Operator_t>&, PsiClassicalFP<1u>& psi, MonteCarlo_tt<PauliString>&);
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalFP<1u>& psi, PsiDeep&, MonteCarlo_tt<PauliString>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator_t&, PsiClassicalFP<1u>&, MonteCarlo_tt<PauliString>&);
template Array<complex_t> ExpectationValue::gradient(const Operator_t&, PsiClassicalFP<1u>& psi, MonteCarlo_tt<PauliString>&);
template pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(const Operator_t&, PsiClassicalFP<1u>& psi, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalFP<2u>& psi, MonteCarlo_tt<PauliString>&);
template Array<complex_t> ExpectationValue::operator()(const vector<Operator_t>&, PsiClassicalFP<2u>& psi, MonteCarlo_tt<PauliString>&);
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalFP<2u>& psi, PsiDeep&, MonteCarlo_tt<PauliString>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator_t&, PsiClassicalFP<2u>&, MonteCarlo_tt<PauliString>&);
template Array<complex_t> ExpectationValue::gradient(const Operator_t&, PsiClassicalFP<2u>& psi, MonteCarlo_tt<PauliString>&);
template pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(const Operator_t&, PsiClassicalFP<2u>& psi, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalANN<1u>& psi, MonteCarlo_tt<PauliString>&);
template Array<complex_t> ExpectationValue::operator()(const vector<Operator_t>&, PsiClassicalANN<1u>& psi, MonteCarlo_tt<PauliString>&);
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalANN<1u>& psi, PsiDeep&, MonteCarlo_tt<PauliString>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator_t&, PsiClassicalANN<1u>&, MonteCarlo_tt<PauliString>&);
template Array<complex_t> ExpectationValue::gradient(const Operator_t&, PsiClassicalANN<1u>& psi, MonteCarlo_tt<PauliString>&);
template pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(const Operator_t&, PsiClassicalANN<1u>& psi, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalANN<2u>& psi, MonteCarlo_tt<PauliString>&);
template Array<complex_t> ExpectationValue::operator()(const vector<Operator_t>&, PsiClassicalANN<2u>& psi, MonteCarlo_tt<PauliString>&);
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalANN<2u>& psi, PsiDeep&, MonteCarlo_tt<PauliString>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator_t&, PsiClassicalANN<2u>&, MonteCarlo_tt<PauliString>&);
template Array<complex_t> ExpectationValue::gradient(const Operator_t&, PsiClassicalANN<2u>& psi, MonteCarlo_tt<PauliString>&);
template pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(const Operator_t&, PsiClassicalANN<2u>& psi, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS)
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiDeep& psi, ExactSummation_t<Spins>&);
template Array<complex_t> ExpectationValue::operator()(const vector<Operator_t>&, PsiDeep& psi, ExactSummation_t<Spins>&);
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiDeep& psi, PsiDeep&, ExactSummation_t<Spins>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator_t&, PsiDeep&, ExactSummation_t<Spins>&);
template Array<complex_t> ExpectationValue::gradient(const Operator_t&, PsiDeep& psi, ExactSummation_t<Spins>&);
template pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(const Operator_t&, PsiDeep& psi, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiFullyPolarized& psi, ExactSummation_t<Spins>&);
template Array<complex_t> ExpectationValue::operator()(const vector<Operator_t>&, PsiFullyPolarized& psi, ExactSummation_t<Spins>&);
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiFullyPolarized& psi, PsiDeep&, ExactSummation_t<Spins>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator_t&, PsiFullyPolarized&, ExactSummation_t<Spins>&);
template Array<complex_t> ExpectationValue::gradient(const Operator_t&, PsiFullyPolarized& psi, ExactSummation_t<Spins>&);
template pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(const Operator_t&, PsiFullyPolarized& psi, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalFP<1u>& psi, ExactSummation_t<Spins>&);
template Array<complex_t> ExpectationValue::operator()(const vector<Operator_t>&, PsiClassicalFP<1u>& psi, ExactSummation_t<Spins>&);
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalFP<1u>& psi, PsiDeep&, ExactSummation_t<Spins>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator_t&, PsiClassicalFP<1u>&, ExactSummation_t<Spins>&);
template Array<complex_t> ExpectationValue::gradient(const Operator_t&, PsiClassicalFP<1u>& psi, ExactSummation_t<Spins>&);
template pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(const Operator_t&, PsiClassicalFP<1u>& psi, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalFP<2u>& psi, ExactSummation_t<Spins>&);
template Array<complex_t> ExpectationValue::operator()(const vector<Operator_t>&, PsiClassicalFP<2u>& psi, ExactSummation_t<Spins>&);
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalFP<2u>& psi, PsiDeep&, ExactSummation_t<Spins>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator_t&, PsiClassicalFP<2u>&, ExactSummation_t<Spins>&);
template Array<complex_t> ExpectationValue::gradient(const Operator_t&, PsiClassicalFP<2u>& psi, ExactSummation_t<Spins>&);
template pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(const Operator_t&, PsiClassicalFP<2u>& psi, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalANN<1u>& psi, ExactSummation_t<Spins>&);
template Array<complex_t> ExpectationValue::operator()(const vector<Operator_t>&, PsiClassicalANN<1u>& psi, ExactSummation_t<Spins>&);
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalANN<1u>& psi, PsiDeep&, ExactSummation_t<Spins>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator_t&, PsiClassicalANN<1u>&, ExactSummation_t<Spins>&);
template Array<complex_t> ExpectationValue::gradient(const Operator_t&, PsiClassicalANN<1u>& psi, ExactSummation_t<Spins>&);
template pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(const Operator_t&, PsiClassicalANN<1u>& psi, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalANN<2u>& psi, ExactSummation_t<Spins>&);
template Array<complex_t> ExpectationValue::operator()(const vector<Operator_t>&, PsiClassicalANN<2u>& psi, ExactSummation_t<Spins>&);
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalANN<2u>& psi, PsiDeep&, ExactSummation_t<Spins>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator_t&, PsiClassicalANN<2u>&, ExactSummation_t<Spins>&);
template Array<complex_t> ExpectationValue::gradient(const Operator_t&, PsiClassicalANN<2u>& psi, ExactSummation_t<Spins>&);
template pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(const Operator_t&, PsiClassicalANN<2u>& psi, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS)
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiDeep& psi, ExactSummation_t<PauliString>&);
template Array<complex_t> ExpectationValue::operator()(const vector<Operator_t>&, PsiDeep& psi, ExactSummation_t<PauliString>&);
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiDeep& psi, PsiDeep&, ExactSummation_t<PauliString>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator_t&, PsiDeep&, ExactSummation_t<PauliString>&);
template Array<complex_t> ExpectationValue::gradient(const Operator_t&, PsiDeep& psi, ExactSummation_t<PauliString>&);
template pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(const Operator_t&, PsiDeep& psi, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiFullyPolarized& psi, ExactSummation_t<PauliString>&);
template Array<complex_t> ExpectationValue::operator()(const vector<Operator_t>&, PsiFullyPolarized& psi, ExactSummation_t<PauliString>&);
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiFullyPolarized& psi, PsiDeep&, ExactSummation_t<PauliString>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator_t&, PsiFullyPolarized&, ExactSummation_t<PauliString>&);
template Array<complex_t> ExpectationValue::gradient(const Operator_t&, PsiFullyPolarized& psi, ExactSummation_t<PauliString>&);
template pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(const Operator_t&, PsiFullyPolarized& psi, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalFP<1u>& psi, ExactSummation_t<PauliString>&);
template Array<complex_t> ExpectationValue::operator()(const vector<Operator_t>&, PsiClassicalFP<1u>& psi, ExactSummation_t<PauliString>&);
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalFP<1u>& psi, PsiDeep&, ExactSummation_t<PauliString>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator_t&, PsiClassicalFP<1u>&, ExactSummation_t<PauliString>&);
template Array<complex_t> ExpectationValue::gradient(const Operator_t&, PsiClassicalFP<1u>& psi, ExactSummation_t<PauliString>&);
template pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(const Operator_t&, PsiClassicalFP<1u>& psi, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalFP<2u>& psi, ExactSummation_t<PauliString>&);
template Array<complex_t> ExpectationValue::operator()(const vector<Operator_t>&, PsiClassicalFP<2u>& psi, ExactSummation_t<PauliString>&);
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalFP<2u>& psi, PsiDeep&, ExactSummation_t<PauliString>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator_t&, PsiClassicalFP<2u>&, ExactSummation_t<PauliString>&);
template Array<complex_t> ExpectationValue::gradient(const Operator_t&, PsiClassicalFP<2u>& psi, ExactSummation_t<PauliString>&);
template pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(const Operator_t&, PsiClassicalFP<2u>& psi, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalANN<1u>& psi, ExactSummation_t<PauliString>&);
template Array<complex_t> ExpectationValue::operator()(const vector<Operator_t>&, PsiClassicalANN<1u>& psi, ExactSummation_t<PauliString>&);
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalANN<1u>& psi, PsiDeep&, ExactSummation_t<PauliString>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator_t&, PsiClassicalANN<1u>&, ExactSummation_t<PauliString>&);
template Array<complex_t> ExpectationValue::gradient(const Operator_t&, PsiClassicalANN<1u>& psi, ExactSummation_t<PauliString>&);
template pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(const Operator_t&, PsiClassicalANN<1u>& psi, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalANN<2u>& psi, ExactSummation_t<PauliString>&);
template Array<complex_t> ExpectationValue::operator()(const vector<Operator_t>&, PsiClassicalANN<2u>& psi, ExactSummation_t<PauliString>&);
template complex<double> ExpectationValue::operator()(const Operator_t&, PsiClassicalANN<2u>& psi, PsiDeep&, ExactSummation_t<PauliString>&);
template pair<double, complex<double>> ExpectationValue::fluctuation(const Operator_t&, PsiClassicalANN<2u>&, ExactSummation_t<PauliString>&);
template Array<complex_t> ExpectationValue::gradient(const Operator_t&, PsiClassicalANN<2u>& psi, ExactSummation_t<PauliString>&);
template pair<Array<complex_t>, Array<double>> ExpectationValue::gradient_with_noise(const Operator_t&, PsiClassicalANN<2u>& psi, ExactSummation_t<PauliString>&);
#endif

} // namespace ann_on_gpu

#endif // LEAN_AND_MEAN
