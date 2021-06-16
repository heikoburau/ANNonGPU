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


#define TILE_SIZE 128


template<typename Psi_t, typename Ensemble>
void TDVP::compute_averages(const Operator_t& op, Psi_t& psi, Ensemble& ensemble, true_t) {
    auto num_params = this->num_params;
    auto op_kernel = op.kernel();
    auto psi_kernel = psi.kernel();
    auto E_local_ptr = this->E_local.data();
    auto E2_local_ptr = this->E2_local.data();
    auto O_k_ptr = this->O_k_ar.data();
    auto S_ptr = this->S_matrix.data();
    auto F_ptr = this->F_vector.data();
    auto O_k_samples_ptr = this->O_k_samples->data();
    auto weight_samples_ptr = this->weight_samples->data();
    auto total_weight_ptr = this->total_weight.data();

    using PsiRef = typename Psi_t::PsiRef;

    ensemble.foreach(
        psi.psi_ref,
        [=] __device__ __host__ (
            const unsigned int index,
            const typename Ensemble::Basis_t& configuration,
            const typename PsiRef::dtype& log_psi_ref,
            typename PsiRef::Payload& payload_ref,
            const typename PsiRef::real_dtype weight
        ) {
            #include "cuda_kernel_defines.h"

            SHARED complex_t                   log_psi;
            SHARED typename Psi_t::Payload     payload;

            psi_kernel.init_payload(payload, configuration, index);
            psi_kernel.log_psi_s(log_psi, configuration, payload);
            SYNC;

            SHARED complex_t local_energy;
            op_kernel.local_energy(local_energy, psi_kernel, configuration, log_psi, payload);

            SHARED double prob_ratio;

            SINGLE {
                prob_ratio = exp(2.0 * (log_psi.real() - log_psi_ref.real()));
                generic_atomicAdd(total_weight_ptr, prob_ratio * weight);
                generic_atomicAdd(E_local_ptr, prob_ratio * weight * local_energy);
                generic_atomicAdd(E2_local_ptr, prob_ratio * weight * abs2(local_energy));

                weight_samples_ptr[index] = prob_ratio * weight;
            }

            psi_kernel.init_payload(payload, configuration, index);
            psi_kernel.foreach_O_k(
                configuration,
                payload,
                [&](const unsigned int k, const complex_t& O_k) {
                    generic_atomicAdd(&O_k_ptr[k], prob_ratio * weight * O_k);
                    generic_atomicAdd(&F_ptr[k], prob_ratio * weight * local_energy * conj(O_k));
                    generic_atomicAdd(&O_k_samples_ptr[index * num_params + k], O_k);
                }
            );
        }
    );
}


template<typename Psi_t, typename Ensemble>
void TDVP::compute_averages(const Operator_t& op, Psi_t& psi, Ensemble& ensemble, false_t) {
    auto num_params = this->F_vector.size();
    auto op_kernel = op.kernel();
    auto psi_kernel = psi.kernel();
    auto E_local_ptr = this->E_local.data();
    auto E2_local_ptr = this->E2_local.data();
    auto O_k_ptr = this->O_k_ar.data();
    auto S_ptr = this->S_matrix.data();
    auto F_ptr = this->F_vector.data();
    auto O_k_samples_ptr = this->O_k_samples->data();
    // auto E_local_samples_ptr = this->E_local_samples->data();
    auto weight_samples_ptr = this->weight_samples->data();
    // auto total_weight_ptr = this->total_weight.data();

    ensemble.foreach(
        psi,
        [=] __device__ __host__ (
            const unsigned int index,
            const typename Ensemble::Basis_t& configuration,
            const typename Psi_t::dtype& log_psi,
            typename Psi_t::Payload& payload,
            const typename Psi_t::real_dtype weight
        ) {
            #include "cuda_kernel_defines.h"

            SHARED complex_t local_energy;
            op_kernel.local_energy(local_energy, psi_kernel, configuration, log_psi, payload);

            SINGLE {
                // generic_atomicAdd(total_weight_ptr, weight);
                generic_atomicAdd(E_local_ptr, weight * local_energy);
                generic_atomicAdd(E2_local_ptr, weight * abs2(local_energy));

                weight_samples_ptr[index] = weight;
                // E_local_samples_ptr[index] = local_energy;
            }

            psi_kernel.init_payload(payload, configuration, index);
            psi_kernel.foreach_O_k(
                configuration,
                payload,
                [&](const unsigned int k, const complex_t& O_k) {
                    generic_atomicAdd(&O_k_ptr[k], weight * O_k);
                    generic_atomicAdd(&F_ptr[k], weight * local_energy * conj(O_k));
                    O_k_samples_ptr[index * num_params + k] = O_k;
                }
            );
        }
    );
}

template<typename Psi_t, typename Ensemble>
void TDVP::compute_averages_fast(const Operator_t& op, Psi_t& psi, Ensemble& ensemble) {
    auto num_params = this->F_vector.size();
    auto op_kernel = op.kernel();
    auto psi_kernel = psi.kernel();
    auto E_local_ptr = this->E_local.data();
    auto E2_local_ptr = this->E2_local.data();
    auto O_k_ptr = this->O_k_ar.data();
    auto S_ptr = this->S_matrix.data();
    auto F_ptr = this->F_vector.data();

    ensemble.foreach(
        psi,
        [=] __device__ __host__ (
            const unsigned int index,
            const typename Ensemble::Basis_t& configuration,
            const typename Psi_t::dtype& log_psi,
            typename Psi_t::Payload& payload,
            const typename Psi_t::real_dtype weight
        ) {
            #include "cuda_kernel_defines.h"

            SHARED complex_t local_energy;
            op_kernel.local_energy(local_energy, psi_kernel, configuration, log_psi, payload);

            SINGLE {
                generic_atomicAdd(E_local_ptr, weight * local_energy);
                generic_atomicAdd(E2_local_ptr, weight * abs2(local_energy));
            }

            psi_kernel.foreach_O_k(
                configuration,
                payload,
                [&](const unsigned int k, const complex_t& O_k) {
                    generic_atomicAdd(&O_k_ptr[k], weight * O_k);
                    generic_atomicAdd(&F_ptr[k], weight * local_energy * conj(O_k));

                    for(auto k_prime = 0u; k_prime < num_params; k_prime++) {
                        generic_atomicAdd(
                            &S_ptr[k * num_params + k_prime],
                            weight * conj(O_k) * psi_kernel.get_O_k(k_prime, configuration, payload)
                        );
                    }
                }
            );
        }
    );
}


template<typename Psi_t, typename Ensemble, typename use_psi_ref_t>
void TDVP::eval(const Operator_t& op, Psi_t& psi, Ensemble& ensemble, use_psi_ref_t use_psi_ref) {
    if(!this->O_k_samples || this->O_k_samples->size() != ensemble.get_num_steps() * psi.num_params) {
        this->O_k_samples = unique_ptr<Array<complex_t>>(
            new Array<complex_t>(ensemble.get_num_steps() * psi.num_params, psi.gpu)
        );
    }
    // if(!this->E_local_samples || this->E_local_samples->size() != ensemble.get_num_steps()) {
    //     this->E_local_samples = unique_ptr<Array<complex_t>>(
    //         new Array<complex_t>(ensemble.get_num_steps(), psi.gpu)
    //     );
    // }
    if(!this->weight_samples || this->weight_samples->size() != ensemble.get_num_steps()) {
        this->weight_samples = unique_ptr<Array<double>>(
            new Array<double>(ensemble.get_num_steps(), psi.gpu)
        );
    }

    this->E_local.clear();
    this->E2_local.clear();
    this->O_k_ar.clear();
    this->S_matrix.clear();
    this->F_vector.clear();
    // this->O_k_samples->clear();
    // this->E_local_samples->clear();
    // this->total_weight.clear();

    this->compute_averages(op, psi, ensemble, use_psi_ref);

    auto num_params = this->num_params;
    auto S_ptr = this->S_matrix.data();
    auto O_k_samples_ptr = this->O_k_samples->data();
    auto weight_samples_ptr = this->weight_samples->data();


    auto fill_S_matrix = [=] __device__ __host__ (
        const unsigned int index
    ) {
        #include "cuda_kernel_defines.h"

        SHARED complex_t    row_data[TILE_SIZE]; // using a register is not faster
        SHARED complex_t    col_data[TILE_SIZE];
        SHARED complex_t*   O_k;
        SHARED double       weight;

        SINGLE {
            O_k = O_k_samples_ptr + index * num_params;
            weight = weight_samples_ptr[index];
        }
        SYNC;

        SHARED_MEM_LOOP_BEGIN(tile_row, num_params / TILE_SIZE + 1) {

            MULTI(i, TILE_SIZE) {
                const auto i_abs = tile_row * TILE_SIZE + i;

                if(i_abs < num_params) {
                    row_data[i] = O_k[i_abs];
                }
            }

            SHARED_MEM_LOOP_BEGIN(tile_col, num_params / TILE_SIZE + 1) {
                MULTI(i, TILE_SIZE) {
                    const auto i_abs = tile_col * TILE_SIZE + i;

                    if(i_abs < num_params) {
                        col_data[i] = O_k[i_abs];
                    }
                }
                SYNC;

                MULTI(row, TILE_SIZE) {
                    const auto row_abs = tile_row * TILE_SIZE + row;

                    if(row_abs < num_params) {
                        for(auto col = 0u; col < TILE_SIZE; col++) {
                            const auto col_abs = tile_col * TILE_SIZE + col;

                            if(col_abs < num_params) {
                                generic_atomicAdd(
                                    &S_ptr[row_abs * num_params + col_abs],
                                    weight * conj(row_data[row]) * col_data[col]
                                );
                            }
                        }
                    }
                }

                SHARED_MEM_LOOP_END(tile_col);
            }

            SHARED_MEM_LOOP_END(tile_row);
        }
    };

    if(psi.gpu) {
        cuda_kernel<<<ensemble.get_num_steps(), TILE_SIZE>>>(
            [=] __device__ () {fill_S_matrix(blockIdx.x);}
        );
    }
    else {
        for(auto i = 0u; i < ensemble.get_num_steps(); i++) {
            fill_S_matrix(i);
        }
    }

    this->E_local.update_host();
    this->E2_local.update_host();
    this->O_k_ar.update_host();
    this->S_matrix.update_host();
    this->F_vector.update_host();
    // this->total_weight.update_host();

    // this->O_k_samples->update_host();
    // this->E_local_samples->update_host();


    // this->E_local.front() /= this->total_weight.front();
    // this->E2_local.front() /= this->total_weight.front();
    // for(auto k = 0u; k < num_params; k++) {
    //     this->F_vector[k] /= this->total_weight.front();
    //     this->O_k_ar[k] /= this->total_weight.front();

    //     for(auto k_prime = 0u; k_prime < num_params; k_prime++) {
    //         this->S_matrix[k * num_params + k_prime] /= this->total_weight.front();
    //     }
    // }


    for(auto k = 0u; k < num_params; k++) {
        for(auto k_prime = 0u; k_prime < num_params; k_prime++) {
            this->S_matrix[k * num_params + k_prime] -= (
                conj(this->O_k_ar[k]) * this->O_k_ar[k_prime]
            );
        }

        this->F_vector[k] -= this->E_local.front() * conj(this->O_k_ar[k]);
    }
}


template<typename Psi_t, typename Ensemble>
void TDVP::eval_fast(const Operator_t& op, Psi_t& psi, Ensemble& ensemble) {
    this->E_local.clear();
    this->E2_local.clear();
    this->O_k_ar.clear();
    this->S_matrix.clear();
    this->F_vector.clear();

    this->compute_averages_fast(op, psi, ensemble);

    this->E_local.update_host();
    this->E2_local.update_host();
    this->O_k_ar.update_host();
    this->S_matrix.update_host();
    this->F_vector.update_host();

    for(auto k = 0u; k < num_params; k++) {
        for(auto k_prime = 0u; k_prime < num_params; k_prime++) {
            this->S_matrix[k * num_params + k_prime] -= (
                conj(this->O_k_ar[k]) * this->O_k_ar[k_prime]
            );
        }

        this->F_vector[k] -= this->E_local.front() * conj(this->O_k_ar[k]);
    }
}

#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator_t&, PsiClassicalFP<1u>&, MonteCarlo_tt<Spins>&, true_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator_t&, PsiClassicalFP<2u>&, MonteCarlo_tt<Spins>&, true_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator_t&, PsiClassicalANN<1u>&, MonteCarlo_tt<Spins>&, true_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator_t&, PsiClassicalANN<2u>&, MonteCarlo_tt<Spins>&, true_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator_t&, PsiClassicalFP<1u>&, MonteCarlo_tt<PauliString>&, true_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator_t&, PsiClassicalFP<2u>&, MonteCarlo_tt<PauliString>&, true_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator_t&, PsiClassicalANN<1u>&, MonteCarlo_tt<PauliString>&, true_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator_t&, PsiClassicalANN<2u>&, MonteCarlo_tt<PauliString>&, true_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator_t&, PsiClassicalFP<1u>&, ExactSummation_t<Spins>&, true_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator_t&, PsiClassicalFP<2u>&, ExactSummation_t<Spins>&, true_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator_t&, PsiClassicalANN<1u>&, ExactSummation_t<Spins>&, true_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator_t&, PsiClassicalANN<2u>&, ExactSummation_t<Spins>&, true_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator_t&, PsiClassicalFP<1u>&, ExactSummation_t<PauliString>&, true_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator_t&, PsiClassicalFP<2u>&, ExactSummation_t<PauliString>&, true_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator_t&, PsiClassicalANN<1u>&, ExactSummation_t<PauliString>&, true_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator_t&, PsiClassicalANN<2u>&, ExactSummation_t<PauliString>&, true_t);
#endif

#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_DEEP)
template void TDVP::eval(const Operator_t&, PsiDeep&, MonteCarlo_tt<Spins>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_RBM)
template void TDVP::eval(const Operator_t&, PsiRBM&, MonteCarlo_tt<Spins>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CNN)
template void TDVP::eval(const Operator_t&, PsiCNN&, MonteCarlo_tt<Spins>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator_t&, PsiFullyPolarized&, MonteCarlo_tt<Spins>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator_t&, PsiClassicalFP<1u>&, MonteCarlo_tt<Spins>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator_t&, PsiClassicalFP<2u>&, MonteCarlo_tt<Spins>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator_t&, PsiClassicalANN<1u>&, MonteCarlo_tt<Spins>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator_t&, PsiClassicalANN<2u>&, MonteCarlo_tt<Spins>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_DEEP)
template void TDVP::eval(const Operator_t&, PsiDeep&, MonteCarlo_tt<PauliString>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_RBM)
template void TDVP::eval(const Operator_t&, PsiRBM&, MonteCarlo_tt<PauliString>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CNN)
template void TDVP::eval(const Operator_t&, PsiCNN&, MonteCarlo_tt<PauliString>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator_t&, PsiFullyPolarized&, MonteCarlo_tt<PauliString>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator_t&, PsiClassicalFP<1u>&, MonteCarlo_tt<PauliString>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator_t&, PsiClassicalFP<2u>&, MonteCarlo_tt<PauliString>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator_t&, PsiClassicalANN<1u>&, MonteCarlo_tt<PauliString>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator_t&, PsiClassicalANN<2u>&, MonteCarlo_tt<PauliString>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_DEEP)
template void TDVP::eval(const Operator_t&, PsiDeep&, ExactSummation_t<Spins>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_RBM)
template void TDVP::eval(const Operator_t&, PsiRBM&, ExactSummation_t<Spins>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CNN)
template void TDVP::eval(const Operator_t&, PsiCNN&, ExactSummation_t<Spins>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator_t&, PsiFullyPolarized&, ExactSummation_t<Spins>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator_t&, PsiClassicalFP<1u>&, ExactSummation_t<Spins>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator_t&, PsiClassicalFP<2u>&, ExactSummation_t<Spins>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator_t&, PsiClassicalANN<1u>&, ExactSummation_t<Spins>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator_t&, PsiClassicalANN<2u>&, ExactSummation_t<Spins>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_DEEP)
template void TDVP::eval(const Operator_t&, PsiDeep&, ExactSummation_t<PauliString>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_RBM)
template void TDVP::eval(const Operator_t&, PsiRBM&, ExactSummation_t<PauliString>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CNN)
template void TDVP::eval(const Operator_t&, PsiCNN&, ExactSummation_t<PauliString>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator_t&, PsiFullyPolarized&, ExactSummation_t<PauliString>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator_t&, PsiClassicalFP<1u>&, ExactSummation_t<PauliString>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator_t&, PsiClassicalFP<2u>&, ExactSummation_t<PauliString>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator_t&, PsiClassicalANN<1u>&, ExactSummation_t<PauliString>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator_t&, PsiClassicalANN<2u>&, ExactSummation_t<PauliString>&, false_t);
#endif

} // namespace ann_on_gpu


#endif // LEAN_AND_MEAN
