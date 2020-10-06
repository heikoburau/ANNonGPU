#pragma once

#include "operator/Operator.hpp"
#include "Array.hpp"
#include "operator/Spins.h"
#include "cuda_complex.hpp"
#include "types.h"

#include <memory>
#include <cmath>


namespace ann_on_gpu {

namespace kernel {

struct ExactSummation {
    using Basis_t = Spins;


    unsigned int  num_spin_configurations;
    bool          has_total_z_symmetry;
    Spins*        allowed_spin_configurations;


    inline unsigned int get_num_steps() const {
        return this->num_spin_configurations;
    }

    inline bool has_weights() const {
        return true;
    }

#ifdef __CUDACC__

    template<typename Psi_t, typename Function>
    HDINLINE
    void kernel_foreach(Psi_t psi, Function function) const {
        #include "cuda_kernel_defines.h"

        SHARED Spins        spins;
        SHARED complex_t    log_psi;
        SHARED double       weight;

        SHARED typename Psi_t::dtype angles[Psi_t::max_width];
        SHARED typename Psi_t::dtype activations[Psi_t::max_width];

        #ifdef __CUDA_ARCH__
        const auto spin_index = blockIdx.x;
        #else
        for(auto spin_index = 0u; spin_index < this->num_spin_configurations; spin_index++)
        #endif
        {
            if(this->has_total_z_symmetry) {
                spins = Spins(this->allowed_spin_configurations[spin_index]);
            }
            else {
                spins = Spins(spin_index, psi.get_num_input_units());
            }

            SYNC;

            psi.compute_angles(angles, spins);
            psi.log_psi_s(log_psi, angles, activations);

            SYNC;

            SINGLE {
                weight = psi.probability_s(log_psi.real());
            }

            SYNC;

            function(spin_index, spins, log_psi, angles, activations, weight);
        }
    }

#endif // __CUDACC__

    ExactSummation kernel() const {
        return *this;
    }

};

} // namespace kernel

struct ExactSummation : public kernel::ExactSummation {
    bool          gpu;
    unsigned int  num_spins;
    unique_ptr<Array<Spins>> allowed_spin_configurations_vec;

    ExactSummation(const unsigned int num_spins, const bool gpu);

    // ExactSummation copy() const {
    //     return *this;
    // }

    void set_total_z_symmetry(const int sector);

#ifdef __CUDACC__
    template<typename Psi_t, typename Function>
    inline void foreach(const Psi_t& psi, const Function& function, const int blockDim=-1) const {
        auto this_kernel = this->kernel();
        const auto psi_kernel = psi.kernel();
        if(this->gpu) {
            const auto blockDim_ = blockDim == -1 ? psi.get_width() : blockDim;

            cuda_kernel<<<this->num_spin_configurations, blockDim_>>>(
                [=] __device__ () {this_kernel.kernel_foreach(psi_kernel, function);}
            );
        }
        else {
            this_kernel.kernel_foreach(psi_kernel, function);
        }
    }
#endif

};

} // namespace ann_on_gpu
