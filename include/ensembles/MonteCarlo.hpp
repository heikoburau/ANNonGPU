#pragma once

#include "operator/Operator.hpp"
#include "policies/Init_Policy.hpp"
#include "policies/Update_Policy.hpp"
#include "bases.hpp"

#ifdef __CUDACC__
    namespace quantum_expression {
        class PauliExpression;
    }
#else
    #include "QuantumExpression/QuantumExpression.hpp"
#endif

#include "RNGStates.hpp"
#include "random.h"
#include "Array.hpp"
#include "cuda_complex.hpp"
#include "types.h"

#include <memory>
#include <random>


namespace ann_on_gpu {

namespace kernel {

template<typename Basis, typename Init_Policy, typename Update_Policy>
struct MonteCarlo_t {
    using Basis_t = Basis;

    RNGStates       rng_states;
    unsigned int    num_samples;
    unsigned int    num_sweeps;
    unsigned int    num_thermalization_sweeps;
    unsigned int    num_markov_chains;
    unsigned int    num_mc_steps_per_chain;

    double          weight;

    Update_Policy   update_policy;

    unsigned int*   acceptances;
    unsigned int*   rejections;


    inline unsigned int get_num_steps() const {
        return this->num_samples;
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

        SHARED Basis_t configuration;

        Init_Policy::call(configuration, psi, &rng_state);
        SYNC;

        SHARED typename Psi_t::dtype        angles[Psi_t::max_width];
        SHARED typename Psi_t::dtype        activations[Psi_t::max_width];
        SHARED typename Psi_t::dtype        log_psi;
        SHARED typename Psi_t::real_dtype   log_psi_real;

        psi.compute_angles(angles, configuration);
        psi.log_psi_s_real(log_psi_real, configuration, angles, activations);

        // thermalization
        SHARED_MEM_LOOP_BEGIN(i, this->num_thermalization_sweeps * psi.get_num_input_units()) {
            this->mc_update(psi, log_psi_real, configuration, &rng_state, angles, activations);

            SHARED_MEM_LOOP_END(i);
        }

        // main loop
        SHARED_MEM_LOOP_BEGIN(mc_step_within_chain, this->num_mc_steps_per_chain) {

            SHARED_MEM_LOOP_BEGIN(
                i,
                this->num_sweeps * psi.get_num_input_units()
            ) {
                this->mc_update(psi, log_psi_real, configuration, &rng_state, angles, activations);

                SHARED_MEM_LOOP_END(i);
            }

            psi.log_psi_s(log_psi, configuration, angles, activations);

            function(
                mc_step_within_chain * this->num_markov_chains + markov_index,
                configuration,
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
    void mc_update(
        const Psi_t& psi,
        typename Psi_t::real_dtype& log_psi_real,
        Basis_t& configuration,
        void* rng_state,
        typename Psi_t::dtype* angles,
        typename Psi_t::dtype* activations
    ) const {
        #include "cuda_kernel_defines.h"
        using real_dtype = typename Psi_t::real_dtype;

        SHARED Basis_t next_configuration;
        this->update_policy(next_configuration, configuration, psi, rng_state);
        psi.update_input_units(angles, configuration, next_configuration);

        SHARED real_dtype next_log_psi_real;
        psi.log_psi_s_real(next_log_psi_real, next_configuration, angles, activations);

        SHARED bool accepted;
        SHARED real_dtype ratio;
        SINGLE {
            ratio = exp(real_dtype(2.0) * (next_log_psi_real - log_psi_real));

            if(ratio > real_dtype(1.0) || real_dtype(random_real(rng_state)) <= ratio) {
                log_psi_real = next_log_psi_real;
                configuration = next_configuration;
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
            psi.update_input_units(angles, next_configuration, configuration);
        }

    }

#endif // __CUDACC__


    inline MonteCarlo_t& kernel() {
        return *this;
    }

    inline const MonteCarlo_t& kernel() const {
        return *this;
    }
};

} // namespace kernel


template<typename Basis_t, typename Init_Policy, typename Update_Policy>
struct MonteCarlo_t : public kernel::MonteCarlo_t<Basis_t, Init_Policy, typename Update_Policy::kernel_t> {
    bool gpu;

    Array<unsigned int> acceptances_ar;
    Array<unsigned int> rejections_ar;
    RNGStates           rng_states;

    Update_Policy       update_policy;

    MonteCarlo_t(
        const unsigned int  num_samples,
        const unsigned int  num_sweeps,
        const unsigned int  num_thermalization_sweeps,
        const unsigned int  num_markov_chains,
        const Update_Policy update_policy,
        const bool          gpu
    );
    MonteCarlo_t(const MonteCarlo_t& other);

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


template<typename Basis_t>
using MonteCarlo_tt = MonteCarlo_t<Basis_t, Init_Policy<Basis_t>, Update_Policy<Basis_t>>;


#ifdef ENABLE_SPINS
using MonteCarloSpins = MonteCarlo_tt<Spins>;

#ifndef __CUDACC__
inline MonteCarloSpins make_MonteCarloSpins(
    const unsigned int  num_samples,
    const unsigned int  num_sweeps,
    const unsigned int  num_thermalization_sweeps,
    const unsigned int  num_markov_chains,
    const bool          gpu
) {
    return MonteCarloSpins(
        num_samples, num_sweeps, num_thermalization_sweeps, num_markov_chains, Update_Policy<Spins>(), gpu
    );
}
#endif // __CUDACC__

#endif // ENABLE_SPINS


#ifdef ENABLE_PAULIS
using MonteCarloPaulis = MonteCarlo_tt<PauliString>;


#ifndef __CUDACC__
inline MonteCarloPaulis make_MonteCarloPaulis(
    const unsigned int  num_samples,
    const unsigned int  num_sweeps,
    const unsigned int  num_thermalization_sweeps,
    const unsigned int  num_markov_chains,
    const quantum_expression::PauliExpression update_expr,
    const bool          gpu
) {
    return MonteCarloPaulis(
        num_samples,
        num_sweeps,
        num_thermalization_sweeps,
        num_markov_chains,
        Update_Policy<PauliString>(Operator(update_expr, gpu)),
        gpu
    );
}
#endif // __CUDACC__

#endif // ENABLE_PAULIS


} // namespace ann_on_gpu
