#pragma once

#include "operator/Operator.hpp"
#include "operator/Spins.h"
#include "RNGStates.hpp"
#include "random.h"
#include "Array.hpp"
#include "cuda_complex.hpp"
#include "types.h"

#include <memory>
#include <random>


namespace ann_on_gpu {

namespace kernel {

struct MonteCarloLoop {

    using Basis_t = Spins;

    RNGStates       rng_states;
    unsigned int    num_samples;
    unsigned int    num_sweeps;
    unsigned int    num_thermalization_sweeps;
    unsigned int    num_markov_chains;

    bool            has_total_z_symmetry;
    int             symmetry_sector;
    double          weight;

    unsigned int    num_mc_steps_per_chain;

    bool            fast_sweep;
    unsigned int    fast_sweep_num_tries;

    unsigned int*   acceptances;
    unsigned int*   rejections;


    inline unsigned int get_num_steps() const {
        return this->num_samples;
    }

    inline bool has_weights() const {
        return false;
    }

#ifdef __CUDACC__

    template<bool total_z_symmetry, typename Psi_t, typename Function>
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
            mt19937 rng_state;
        #endif
        SINGLE {
            this->rng_states.get_state(rng_state, markov_index);
        }

        SHARED Spins spins;

        SINGLE {
            if(total_z_symmetry) {
                #if (MAX_SPINS <= 64)
                spins = Spins(
                    random_n_over_k_bitstring(
                        psi.get_num_input_units(),
                        (this->symmetry_sector + psi.get_num_input_units()) / 2,
                        &rng_state
                    ),
                    psi.get_num_input_units()
                );
                #endif
            }
            else {
                spins.set_randomly(&rng_state, psi.get_num_input_units());
            }
        }
        SYNC;

        SHARED typename Psi_t::dtype        angles[Psi_t::max_width];
        SHARED typename Psi_t::dtype        activations[Psi_t::max_width];
        SHARED typename Psi_t::dtype        log_psi;
        SHARED typename Psi_t::real_dtype   log_psi_real;

        psi.compute_angles(angles, spins);
        psi.log_psi_s_real(log_psi_real, spins, angles, activations);

        this->thermalize<total_z_symmetry>(psi, log_psi_real, spins, &rng_state, angles, activations);

        SHARED_MEM_LOOP_BEGIN(mc_step_within_chain, this->num_mc_steps_per_chain) {

            SHARED_MEM_LOOP_BEGIN(
                i,
                this->num_sweeps * (this->fast_sweep ? 1u : psi.get_num_input_units())
            ) {
                this->mc_update<total_z_symmetry>(psi, log_psi_real, spins, &rng_state, angles, activations);

                SHARED_MEM_LOOP_END(i);
            }

            psi.log_psi_s(log_psi, spins, angles, activations);

            function(
                mc_step_within_chain * this->num_markov_chains + markov_index,
                spins,
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

    template<bool total_z_symmetry, typename Psi_t>
    HDINLINE
    void thermalize(const Psi_t& psi, typename Psi_t::real_dtype& log_psi_real, Spins& spins, void* rng_state, typename Psi_t::dtype* angles, typename Psi_t::dtype* activations) const {
        #include "cuda_kernel_defines.h"

        SHARED_MEM_LOOP_BEGIN(i, this->num_thermalization_sweeps * psi.get_num_input_units()) {
        // for(auto i = 0u; i < this->num_thermalization_sweeps * psi.get_num_input_units(); i++) {
            this->mc_update<total_z_symmetry>(psi, log_psi_real, spins, rng_state, angles, activations);

            SHARED_MEM_LOOP_END(i);
        }
    }

    template<bool total_z_symmetry, typename Psi_t>
    HDINLINE
    void mc_update(
        const Psi_t& psi,
        typename Psi_t::real_dtype& log_psi_real,
        Spins& spins, void* rng_state,
        typename Psi_t::dtype* angles,
        typename Psi_t::dtype* activations
    ) const {
        #include "cuda_kernel_defines.h"
        using real_dtype = typename Psi_t::real_dtype;

        SHARED int position;
        SHARED int second_position;
        SHARED Spins original_spins;

        SINGLE {
            original_spins = spins;
            position = random_uint32(rng_state) % psi.get_num_input_units();
            spins.flip(position);
        }

        if(total_z_symmetry) {
            SINGLE {
                while(true) {
                    second_position = random_uint32(rng_state) % psi.get_num_input_units();
                    if(spins[second_position] == spins[position]) {
                        spins.flip(second_position);
                        break;
                    }
                }
            }
        }
        psi.update_input_units(angles, spins, original_spins);

        SHARED real_dtype next_log_psi_real;
        psi.log_psi_s_real(next_log_psi_real, spins, angles, activations);

        SHARED bool accepted;
        SHARED real_dtype ratio;
        SINGLE {
            ratio = exp(real_dtype(2.0) * (next_log_psi_real - log_psi_real));
            // printf("%f\n", ratio);

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
            // flip back spin(s)

            psi.update_input_units(angles, original_spins, spins);
            SINGLE {
                spins = original_spins;
            }
        }
    }

#endif // __CUDACC__

};

} // namespace kernel


struct MonteCarloLoop : public kernel::MonteCarloLoop {
    bool gpu;

    Array<unsigned int> acceptances_ar;
    Array<unsigned int> rejections_ar;
    RNGStates           rng_states;

    MonteCarloLoop(
        const unsigned int num_samples,
        const unsigned int num_sweeps,
        const unsigned int num_thermalization_sweeps,
        const unsigned int num_markov_chains,
        const bool         gpu
    );
    MonteCarloLoop(MonteCarloLoop& other);

    inline void set_total_z_symmetry(const int sector) {
        this->symmetry_sector = sector;
        this->has_total_z_symmetry = true;
    }

    inline void set_fast_sweep(const unsigned int num_tries) {
        this->fast_sweep = true;
        this->fast_sweep_num_tries = num_tries;
    }

#ifdef __CUDACC__
    template<typename Psi_t, typename Function>
    inline void foreach(const Psi_t& psi, const Function& function, const int blockDim=-1) {
        auto this_kernel = this->kernel();
        auto psi_kernel = psi.kernel();

        this->acceptances_ar.clear();
        this->rejections_ar.clear();

        #ifdef TIMING
            const auto begin = clock::now();
        #endif

        if(this->gpu) {
            const auto blockDim_ = blockDim == -1 ? psi.get_width() : blockDim;

            if(this->has_total_z_symmetry) {
                cuda_kernel<<<this->num_markov_chains, blockDim_>>>(
                    [=] __device__ () {this_kernel.kernel_foreach<true>(psi_kernel, function);}
                );
            }
            else {
                cuda_kernel<<<this->num_markov_chains, blockDim_>>>(
                    [=] __device__ () {this_kernel.kernel_foreach<false>(psi_kernel, function);}
                );
            }
        }
        else {
            if(this->has_total_z_symmetry) {
                this_kernel.kernel_foreach<true>(psi_kernel, function);
            }
            else {
                this_kernel.kernel_foreach<false>(psi_kernel, function);
            }
        }

        this->acceptances_ar.update_host();
        this->rejections_ar.update_host();

        #ifdef TIMING
            if(this->gpu) {
                cudaDeviceSynchronize();
            }
            const auto end = clock::now();
            log_duration("MonteCarloLoop::foreach", end - begin);
        #endif
    }
#endif

    inline kernel::MonteCarloLoop kernel() const {
        return static_cast<kernel::MonteCarloLoop>(*this);
    }
};


} // namespace ann_on_gpu
