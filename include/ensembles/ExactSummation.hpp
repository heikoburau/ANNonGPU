#pragma once

#include "operator/Operator.hpp"
#include "bases.hpp"
#include "Array.hpp"
#include "cuda_complex.hpp"
#include "types.h"

#include <memory>
#include <cmath>


namespace ann_on_gpu {

namespace kernel {

template<typename Basis>
struct ExactSummation_t {
    using Basis_t = Basis;

    unsigned int  num_sites;
    unsigned int  num_configurations;
    // bool          has_total_z_symmetry;
    // Basis_t*        allowed_spin_configurations;


    inline unsigned int get_num_steps() const {
        return this->num_configurations;
    }

#ifdef __CUDACC__

    template<typename Psi_t, typename Function>
    HDINLINE
    void kernel_foreach(Psi_t psi, Function function) const {
        #include "cuda_kernel_defines.h"

        SHARED Basis_t                  configuration;
        SHARED typename Psi_t::dtype    log_psi;
        SHARED double                   weight;

        SHARED typename Psi_t::Payload payload;

        #ifdef __CUDA_ARCH__
        const auto conf_index = blockIdx.x;
        #else
        for(auto conf_index = 0u; conf_index < this->num_configurations; conf_index++)
        #endif
        {
            // if(this->has_total_z_symmetry) {
            //     configuration = Basis_t(this->allowed_spin_configurations[conf_index]);
            // }
            // else {
            configuration = Basis_t::enumerate(conf_index);
            // }

            SYNC;

            psi.init_payload(payload, configuration, conf_index);
            psi.log_psi_s(log_psi, configuration, payload);

            SYNC;

            SINGLE {
                weight = exp(2.0 * log_psi.real());
            }

            SYNC;

            function(conf_index, configuration, log_psi, payload, weight);
        }
    }

#endif // __CUDACC__

    ExactSummation_t kernel() const {
        return *this;
    }

};

} // namespace kernel

template<typename Basis_t>
struct ExactSummation_t : public kernel::ExactSummation_t<Basis_t> {
    bool gpu;

    // unique_ptr<Array<Basis_t>> allowed_spin_configurations_vec;

    ExactSummation_t(const unsigned int num_sites, const bool gpu);

    // ExactSummation_t copy() const {
    //     return *this;
    // }

    // void set_total_z_symmetry(const int sector);

#ifdef __CUDACC__
    template<typename Psi_t, typename Function>
    inline void foreach(Psi_t& psi, const Function& function, const int blockDim=-1) const {
        auto this_kernel = this->kernel();
        const auto psi_kernel = psi.kernel();

        if(this->gpu) {
            const auto blockDim_ = blockDim == -1 ? psi.get_width() : blockDim;

            cuda_kernel<<<this->num_configurations, blockDim_>>>(
                [=] __device__ () {this_kernel.kernel_foreach(psi_kernel, function);}
            );
        }
        else {
            this_kernel.kernel_foreach(psi_kernel, function);
        }
    }
#endif

};


#ifdef ENABLE_SPINS
using ExactSummationSpins = ExactSummation_t<Spins>;
#endif  // ENABLE_SPINS

#ifdef ENABLE_PAULIS
using ExactSummationPaulis = ExactSummation_t<PauliString>;
#endif  // ENABLE_PAULIS

#ifdef ENABLE_FERMIONS
using ExactSummationFermions = ExactSummation_t<Fermions>;
#endif  // ENABLE_FERMIONS


} // namespace ann_on_gpu
