#pragma once

#include "operator/Operator.hpp"
#include "operator/PauliString.hpp"
#include "RNGStates.hpp"
#include "random.h"
#include "Array.hpp"
#include "cuda_complex.hpp"
#include "types.h"

#include <memory>
#include <random>


namespace ann_on_gpu {

namespace kernel {

struct MonteCarloLoopPaulis {

    using Basis_t = PauliString;

    RNGStates       rng_states;
    unsigned int    num_samples;
    unsigned int    num_sweeps;
    unsigned int    num_thermalization_sweeps;
    unsigned int    num_markov_chains;
    unsigned int    num_mc_steps_per_chain;

    double          weight;

    Operator        update_operator;

    unsigned int*   acceptances;
    unsigned int*   rejections;


    inline unsigned int get_num_steps() const {
        return this->num_samples;
    }

    inline bool has_weights() const {
        return false;
    }

#ifdef __CUDACC__

    template<typename Psi_t, typename Function>
    HDINLINE
    void kernel_foreach(const Psi_t psi, Function function) const {
        // ##################################################################################
        //
        // Call with gridDim.x = number of markov chains, blockDim.x = number of hidden spins
        //
        // ##################################################################################
        #include "cuda_kernel_defines.h"

        SHARED unsigned int markov_index;
        #ifdef __CUDA_ARCH__
            __shared__ curandState_t rng_state;
            markov_index = blockIdx.x;
        #else
            markov_index = 0u;
            std::mt19937 rng_state;
        #endif
        SINGLE {
            this->rng_states.get_state(rng_state, markov_index);
        }

        SHARED PauliString pauli_string;

        SINGLE {
            pauli_string = PauliString(0lu, 0lu);
        }
        SYNC;

        SHARED typename Psi_t::dtype        angles[Psi_t::max_width];
        SHARED typename Psi_t::dtype        activations[Psi_t::max_width];
        SHARED typename Psi_t::dtype        log_psi;
        SHARED typename Psi_t::real_dtype   log_psi_real;

        psi.compute_angles(angles, pauli_string);
        psi.log_psi_s_real(log_psi_real, angles, activations);

        this->thermalize(psi, log_psi_real, pauli_string, &rng_state, angles, activations);

        SHARED_MEM_LOOP_BEGIN(mc_step_within_chain, this->num_mc_steps_per_chain) {

            SHARED_MEM_LOOP_BEGIN(
                i,
                this->num_sweeps * psi.get_num_input_units()
            ) {
                this->mc_update(psi, log_psi_real, pauli_string, &rng_state, angles, activations);

                SHARED_MEM_LOOP_END(i);
            }

            psi.log_psi_s(log_psi, angles, activations);

            function(
                mc_step_within_chain * this->num_markov_chains + markov_index,
                pauli_string,
                log_psi,
                angles,
                activations,
                this->weight
            );

            SHARED_MEM_LOOP_END(mc_step_within_chain);
        }

        SINGLE {
            this->rng_states.set_state(rng_state, markov_index);
        }
    }

    template<typename Psi_t>
    HDINLINE
    void thermalize(
        const Psi_t& psi,
        typename Psi_t::real_dtype& log_psi_real,
        PauliString& pauli_string,
        void* rng_state,
        typename Psi_t::dtype* angles,
        typename Psi_t::dtype* activations
    ) const {
        #include "cuda_kernel_defines.h"

        SHARED_MEM_LOOP_BEGIN(i, this->num_thermalization_sweeps * psi.get_num_input_units()) {
            this->mc_update(psi, log_psi_real, pauli_string, rng_state, angles, activations);

            SHARED_MEM_LOOP_END(i);
        }
    }

    template<typename Psi_t>
    HDINLINE
    void mc_update(
        const Psi_t& psi,
        typename Psi_t::real_dtype& log_psi_real,
        PauliString& pauli_string,
        void* rng_state,
        typename Psi_t::dtype* angles,
        typename Psi_t::dtype* activations
    ) const {
        #include "cuda_kernel_defines.h"
        using real_dtype = typename Psi_t::real_dtype;

        SHARED PauliString next_pauli_string;
        SINGLE {
            next_pauli_string = pauli_string.apply(
                this->update_operator.pauli_strings[
                    random_uint32(rng_state) % this->update_operator.num_strings
                ]
            ).vector;
        }
        psi.update_input_units(angles, pauli_string, next_pauli_string);

        SHARED real_dtype next_log_psi_real;
        psi.log_psi_s_real(next_log_psi_real, angles, activations);

        SHARED bool accepted;
        SHARED real_dtype ratio;
        SINGLE {
            ratio = exp(real_dtype(2.0) * (next_log_psi_real - log_psi_real));

            if(ratio > real_dtype(1.0) || real_dtype(random_real(rng_state)) <= ratio) {
                log_psi_real = next_log_psi_real;
                accepted = true;
                generic_atomicAdd(this->acceptances, 1u);
            }
            else {
                accepted = false;
                generic_atomicAdd(this->rejections, 1u);
            }
        }
        SYNC;

        if(!accepted) {
            psi.update_input_units(angles, next_pauli_string, pauli_string);
        }
    }

#endif // __CUDACC__


    inline MonteCarloLoopPaulis& kernel() {
        return *this;
    }

    inline const MonteCarloLoopPaulis& kernel() const {
        return *this;
    }
};

} // namespace kernel


struct MonteCarloLoopPaulis : public kernel::MonteCarloLoopPaulis {
    bool gpu;

    Array<unsigned int> acceptances_ar;
    Array<unsigned int> rejections_ar;
    RNGStates           rng_states;

    Operator update_operator_host;

    MonteCarloLoopPaulis(
        const unsigned int num_samples,
        const unsigned int num_sweeps,
        const unsigned int num_thermalization_sweeps,
        const unsigned int num_markov_chains,
        const Operator     update_operator
    );
    MonteCarloLoopPaulis(MonteCarloLoopPaulis& other);

#ifdef __CUDACC__
    template<typename Psi_t, typename Function>
    inline void foreach(const Psi_t& psi, const Function& function, const int blockDim=-1) {
        auto this_kernel = this->kernel();
        auto psi_kernel = psi.kernel();

        this->acceptances_ar.clear();
        this->rejections_ar.clear();

        if(this->gpu) {
            const auto blockDim_ = blockDim == -1 ? psi.get_width() : blockDim;

            cuda_kernel<<<this->num_markov_chains, blockDim_>>>(
                [=] __device__ () {this_kernel.kernel_foreach(psi_kernel, function);}
            );
        }
        else {
            this_kernel.kernel_foreach(psi_kernel, function);
        }

        this->acceptances_ar.update_host();
        this->rejections_ar.update_host();
    }
#endif
};


} // namespace ann_on_gpu
