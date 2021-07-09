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
void TDVP::compute_averages(const Operator& op, Psi_t& psi, Ensemble& ensemble, true_t) {
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
void TDVP::compute_averages(const Operator& op, Psi_t& psi, Ensemble& ensemble, false_t) {
    auto num_params = this->F_vector.size();
    auto op_kernel = op.kernel();
    auto psi_kernel = psi.kernel();
    auto E_local_ptr = this->E_local.data();
    auto E2_local_ptr = this->E2_local.data();
    auto O_k_ptr = this->O_k_ar.data();
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
                    generic_atomicAdd(&O_k_samples_ptr[index * num_params + k], O_k);
                }
            );
        }
    );
}

template<typename Psi_t, typename Ensemble>
void TDVP::compute_averages_fast(const Operator& op, Psi_t& psi, Ensemble& ensemble) {
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
void TDVP::eval(const Operator& op, Psi_t& psi, Ensemble& ensemble, use_psi_ref_t use_psi_ref) {
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
    this->O_k_samples->clear();
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
void TDVP::eval_F_vector(const Operator& op, Psi_t& psi, Ensemble& ensemble) {
    if(!this->O_k_samples || this->O_k_samples->size() != ensemble.get_num_steps() * psi.num_params) {
        this->O_k_samples = unique_ptr<Array<complex_t>>(
            new Array<complex_t>(ensemble.get_num_steps() * psi.num_params, psi.gpu)
        );
    }
    if(!this->weight_samples || this->weight_samples->size() != ensemble.get_num_steps()) {
        this->weight_samples = unique_ptr<Array<double>>(
            new Array<double>(ensemble.get_num_steps(), psi.gpu)
        );
    }

    this->E_local.clear();
    this->E2_local.clear();
    this->O_k_ar.clear();
    this->F_vector.clear();
    this->O_k_samples->clear();

    this->compute_averages(op, psi, ensemble, false_t());

    this->E_local.update_host();
    this->E2_local.update_host();
    this->O_k_ar.update_host();
    this->F_vector.update_host();

    for(auto k = 0u; k < this->num_params; k++) {
        this->F_vector[k] -= this->E_local.front() * conj(this->O_k_ar[k]);
    }
}

template<typename Ensemble>
void TDVP::S_dot_vector(
    Ensemble& ensemble
) {
    this->output_vector.clear();

    auto num_params = this->num_params;
    auto O_k_samples_ptr = this->O_k_samples->data();
    auto input_vector_ptr = this->input_vector.data();
    auto output_vector_ptr = this->output_vector.data();
    auto weight_samples_ptr = this->weight_samples->data();


    auto fill_output = [=] __device__ __host__ (
        const unsigned int index
    ) {
        #include "cuda_kernel_defines.h"

        SHARED complex_t    row_data[TILE_SIZE]; // using a register is not faster
        SHARED complex_t    col_data[TILE_SIZE];
        SHARED complex_t    row_dot_product[TILE_SIZE];
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
                    row_dot_product[i] = complex_t(0.0);
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
                                row_dot_product[row] += (
                                    col_data[col] * input_vector_ptr[col_abs]
                                );
                            }
                        }
                    }
                }

                SHARED_MEM_LOOP_END(tile_col);
            }

            MULTI(row, TILE_SIZE) {
                const auto row_abs = tile_row * TILE_SIZE + row;

                if(row_abs < num_params) {
                    generic_atomicAdd(
                        &output_vector_ptr[row_abs],
                        weight * conj(row_data[row]) * row_dot_product[row]
                        // weight * input_vector_ptr[row_abs]
                    );
                }
            }

            SHARED_MEM_LOOP_END(tile_row);
        }
    };

    if(this->gpu) {
        cuda_kernel<<<ensemble.get_num_steps(), TILE_SIZE>>>(
            [=] __device__ () {fill_output(blockIdx.x);}
        );
    }
    else {
        for(auto i = 0u; i < ensemble.get_num_steps(); i++) {
            fill_output(i);
        }
    }

    this->output_vector.update_host();

    auto O_k_dot_input = complex_t(0.0);
    for(auto k = 0u; k < num_params; k++) {
        O_k_dot_input += this->O_k_ar[k] * input_vector[k];
    }

    for(auto k = 0u; k < num_params; k++) {
        this->output_vector[k] -= conj(this->O_k_ar[k]) * O_k_dot_input;
    }
}



template<typename Psi_t, typename Ensemble>
void TDVP::eval_fast(const Operator& op, Psi_t& psi, Ensemble& ensemble) {
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
template void TDVP::eval(const Operator&, PsiClassicalFP<1u>&, MonteCarlo_tt<Spins>&, true_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiClassicalFP<2u>&, MonteCarlo_tt<Spins>&, true_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, PsiClassicalANN<1u>&, MonteCarlo_tt<Spins>&, true_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, PsiClassicalANN<2u>&, MonteCarlo_tt<Spins>&, true_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiClassicalFP<1u>&, MonteCarlo_tt<Fermions>&, true_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiClassicalFP<2u>&, MonteCarlo_tt<Fermions>&, true_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, PsiClassicalANN<1u>&, MonteCarlo_tt<Fermions>&, true_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, PsiClassicalANN<2u>&, MonteCarlo_tt<Fermions>&, true_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiClassicalFP<1u>&, MonteCarlo_tt<PauliString>&, true_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiClassicalFP<2u>&, MonteCarlo_tt<PauliString>&, true_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, PsiClassicalANN<1u>&, MonteCarlo_tt<PauliString>&, true_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, PsiClassicalANN<2u>&, MonteCarlo_tt<PauliString>&, true_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiClassicalFP<1u>&, ExactSummation_t<Spins>&, true_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiClassicalFP<2u>&, ExactSummation_t<Spins>&, true_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, PsiClassicalANN<1u>&, ExactSummation_t<Spins>&, true_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, PsiClassicalANN<2u>&, ExactSummation_t<Spins>&, true_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiClassicalFP<1u>&, ExactSummation_t<Fermions>&, true_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiClassicalFP<2u>&, ExactSummation_t<Fermions>&, true_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, PsiClassicalANN<1u>&, ExactSummation_t<Fermions>&, true_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, PsiClassicalANN<2u>&, ExactSummation_t<Fermions>&, true_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiClassicalFP<1u>&, ExactSummation_t<PauliString>&, true_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiClassicalFP<2u>&, ExactSummation_t<PauliString>&, true_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, PsiClassicalANN<1u>&, ExactSummation_t<PauliString>&, true_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, PsiClassicalANN<2u>&, ExactSummation_t<PauliString>&, true_t);
#endif

#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_DEEP)
template void TDVP::eval(const Operator&, PsiDeep&, MonteCarlo_tt<Spins>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_RBM)
template void TDVP::eval(const Operator&, PsiRBM&, MonteCarlo_tt<Spins>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CNN)
template void TDVP::eval(const Operator&, PsiCNN&, MonteCarlo_tt<Spins>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiFullyPolarized&, MonteCarlo_tt<Spins>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiClassicalFP<1u>&, MonteCarlo_tt<Spins>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiClassicalFP<2u>&, MonteCarlo_tt<Spins>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, PsiClassicalANN<1u>&, MonteCarlo_tt<Spins>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, PsiClassicalANN<2u>&, MonteCarlo_tt<Spins>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_DEEP)
template void TDVP::eval(const Operator&, PsiDeep&, MonteCarlo_tt<Fermions>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_RBM)
template void TDVP::eval(const Operator&, PsiRBM&, MonteCarlo_tt<Fermions>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CNN)
template void TDVP::eval(const Operator&, PsiCNN&, MonteCarlo_tt<Fermions>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiFullyPolarized&, MonteCarlo_tt<Fermions>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiClassicalFP<1u>&, MonteCarlo_tt<Fermions>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiClassicalFP<2u>&, MonteCarlo_tt<Fermions>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, PsiClassicalANN<1u>&, MonteCarlo_tt<Fermions>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, PsiClassicalANN<2u>&, MonteCarlo_tt<Fermions>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_DEEP)
template void TDVP::eval(const Operator&, PsiDeep&, MonteCarlo_tt<PauliString>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_RBM)
template void TDVP::eval(const Operator&, PsiRBM&, MonteCarlo_tt<PauliString>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CNN)
template void TDVP::eval(const Operator&, PsiCNN&, MonteCarlo_tt<PauliString>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiFullyPolarized&, MonteCarlo_tt<PauliString>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiClassicalFP<1u>&, MonteCarlo_tt<PauliString>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiClassicalFP<2u>&, MonteCarlo_tt<PauliString>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, PsiClassicalANN<1u>&, MonteCarlo_tt<PauliString>&, false_t);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, PsiClassicalANN<2u>&, MonteCarlo_tt<PauliString>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_DEEP)
template void TDVP::eval(const Operator&, PsiDeep&, ExactSummation_t<Spins>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_RBM)
template void TDVP::eval(const Operator&, PsiRBM&, ExactSummation_t<Spins>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CNN)
template void TDVP::eval(const Operator&, PsiCNN&, ExactSummation_t<Spins>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiFullyPolarized&, ExactSummation_t<Spins>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiClassicalFP<1u>&, ExactSummation_t<Spins>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiClassicalFP<2u>&, ExactSummation_t<Spins>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, PsiClassicalANN<1u>&, ExactSummation_t<Spins>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, PsiClassicalANN<2u>&, ExactSummation_t<Spins>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_DEEP)
template void TDVP::eval(const Operator&, PsiDeep&, ExactSummation_t<Fermions>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_RBM)
template void TDVP::eval(const Operator&, PsiRBM&, ExactSummation_t<Fermions>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CNN)
template void TDVP::eval(const Operator&, PsiCNN&, ExactSummation_t<Fermions>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiFullyPolarized&, ExactSummation_t<Fermions>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiClassicalFP<1u>&, ExactSummation_t<Fermions>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiClassicalFP<2u>&, ExactSummation_t<Fermions>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, PsiClassicalANN<1u>&, ExactSummation_t<Fermions>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, PsiClassicalANN<2u>&, ExactSummation_t<Fermions>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_DEEP)
template void TDVP::eval(const Operator&, PsiDeep&, ExactSummation_t<PauliString>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_RBM)
template void TDVP::eval(const Operator&, PsiRBM&, ExactSummation_t<PauliString>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CNN)
template void TDVP::eval(const Operator&, PsiCNN&, ExactSummation_t<PauliString>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiFullyPolarized&, ExactSummation_t<PauliString>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiClassicalFP<1u>&, ExactSummation_t<PauliString>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval(const Operator&, PsiClassicalFP<2u>&, ExactSummation_t<PauliString>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, PsiClassicalANN<1u>&, ExactSummation_t<PauliString>&, false_t);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval(const Operator&, PsiClassicalANN<2u>&, ExactSummation_t<PauliString>&, false_t);
#endif

#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_DEEP)
template void TDVP::eval_F_vector(const Operator&, PsiDeep&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_RBM)
template void TDVP::eval_F_vector(const Operator&, PsiRBM&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CNN)
template void TDVP::eval_F_vector(const Operator&, PsiCNN&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval_F_vector(const Operator&, PsiFullyPolarized&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval_F_vector(const Operator&, PsiClassicalFP<1u>&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval_F_vector(const Operator&, PsiClassicalFP<2u>&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval_F_vector(const Operator&, PsiClassicalANN<1u>&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval_F_vector(const Operator&, PsiClassicalANN<2u>&, MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_DEEP)
template void TDVP::eval_F_vector(const Operator&, PsiDeep&, MonteCarlo_tt<Fermions>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_RBM)
template void TDVP::eval_F_vector(const Operator&, PsiRBM&, MonteCarlo_tt<Fermions>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CNN)
template void TDVP::eval_F_vector(const Operator&, PsiCNN&, MonteCarlo_tt<Fermions>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval_F_vector(const Operator&, PsiFullyPolarized&, MonteCarlo_tt<Fermions>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval_F_vector(const Operator&, PsiClassicalFP<1u>&, MonteCarlo_tt<Fermions>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval_F_vector(const Operator&, PsiClassicalFP<2u>&, MonteCarlo_tt<Fermions>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval_F_vector(const Operator&, PsiClassicalANN<1u>&, MonteCarlo_tt<Fermions>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval_F_vector(const Operator&, PsiClassicalANN<2u>&, MonteCarlo_tt<Fermions>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_DEEP)
template void TDVP::eval_F_vector(const Operator&, PsiDeep&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_RBM)
template void TDVP::eval_F_vector(const Operator&, PsiRBM&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CNN)
template void TDVP::eval_F_vector(const Operator&, PsiCNN&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval_F_vector(const Operator&, PsiFullyPolarized&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval_F_vector(const Operator&, PsiClassicalFP<1u>&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval_F_vector(const Operator&, PsiClassicalFP<2u>&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval_F_vector(const Operator&, PsiClassicalANN<1u>&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval_F_vector(const Operator&, PsiClassicalANN<2u>&, MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_DEEP)
template void TDVP::eval_F_vector(const Operator&, PsiDeep&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_RBM)
template void TDVP::eval_F_vector(const Operator&, PsiRBM&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CNN)
template void TDVP::eval_F_vector(const Operator&, PsiCNN&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval_F_vector(const Operator&, PsiFullyPolarized&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval_F_vector(const Operator&, PsiClassicalFP<1u>&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval_F_vector(const Operator&, PsiClassicalFP<2u>&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval_F_vector(const Operator&, PsiClassicalANN<1u>&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval_F_vector(const Operator&, PsiClassicalANN<2u>&, ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_DEEP)
template void TDVP::eval_F_vector(const Operator&, PsiDeep&, ExactSummation_t<Fermions>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_RBM)
template void TDVP::eval_F_vector(const Operator&, PsiRBM&, ExactSummation_t<Fermions>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CNN)
template void TDVP::eval_F_vector(const Operator&, PsiCNN&, ExactSummation_t<Fermions>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval_F_vector(const Operator&, PsiFullyPolarized&, ExactSummation_t<Fermions>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval_F_vector(const Operator&, PsiClassicalFP<1u>&, ExactSummation_t<Fermions>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval_F_vector(const Operator&, PsiClassicalFP<2u>&, ExactSummation_t<Fermions>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval_F_vector(const Operator&, PsiClassicalANN<1u>&, ExactSummation_t<Fermions>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval_F_vector(const Operator&, PsiClassicalANN<2u>&, ExactSummation_t<Fermions>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_DEEP)
template void TDVP::eval_F_vector(const Operator&, PsiDeep&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_RBM)
template void TDVP::eval_F_vector(const Operator&, PsiRBM&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CNN)
template void TDVP::eval_F_vector(const Operator&, PsiCNN&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval_F_vector(const Operator&, PsiFullyPolarized&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval_F_vector(const Operator&, PsiClassicalFP<1u>&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL)
template void TDVP::eval_F_vector(const Operator&, PsiClassicalFP<2u>&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval_F_vector(const Operator&, PsiClassicalANN<1u>&, ExactSummation_t<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template void TDVP::eval_F_vector(const Operator&, PsiClassicalANN<2u>&, ExactSummation_t<PauliString>&);
#endif

#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS)
template void TDVP::S_dot_vector(MonteCarlo_tt<Spins>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_FERMIONS)
template void TDVP::S_dot_vector(MonteCarlo_tt<Fermions>&);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS)
template void TDVP::S_dot_vector(MonteCarlo_tt<PauliString>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS)
template void TDVP::S_dot_vector(ExactSummation_t<Spins>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_FERMIONS)
template void TDVP::S_dot_vector(ExactSummation_t<Fermions>&);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS)
template void TDVP::S_dot_vector(ExactSummation_t<PauliString>&);
#endif


} // namespace ann_on_gpu


#endif // LEAN_AND_MEAN
