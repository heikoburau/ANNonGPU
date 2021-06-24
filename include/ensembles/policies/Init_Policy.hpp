#pragma once
#include "bases.hpp"
#include "types.h"


namespace ann_on_gpu {


template<typename Basis_t>
struct Init_Policy;


#ifdef ENABLE_SPINS

template<>
struct Init_Policy<Spins> {
    template<typename Psi_t>
    static HDINLINE void call(Spins& result, const Psi_t& psi, void* rng_state) {
        #include "cuda_kernel_defines.h"

        SINGLE {
            result.set_randomly(rng_state, psi.num_sites);
        }
    }
};

#endif // ENABLE_SPINS


#ifdef ENABLE_PAULIS

template<>
struct Init_Policy<PauliString> {
    template<typename Psi_t>
    static HDINLINE void call(PauliString& result, const Psi_t& psi, void* rng_state) {
        #include "cuda_kernel_defines.h"

        SINGLE {
            result.set_randomly(rng_state, psi.num_sites);
        }
    }
};

#endif // ENABLE_PAULIS

#ifdef ENABLE_FERMIONS

template<>
struct Init_Policy<Fermions> {
    template<typename Psi_t>
    static HDINLINE void call(Fermions& result, const Psi_t& psi, void* rng_state) {
        #include "cuda_kernel_defines.h"

        SINGLE {
            result.set_randomly(rng_state, psi.num_sites);
        }
    }
};

#endif // ENABLE_FERMIONS


}  // namespace ann_on_gpu
