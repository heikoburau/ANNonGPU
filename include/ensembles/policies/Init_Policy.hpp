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
            result.set_randomly(rng_state, psi.get_num_input_units());
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
            result = PauliString(0lu, 0lu);
        }
    }
};

#endif // ENABLE_PAULIS


}  // namespace ann_on_gpu
