#pragma once
#include "bases.hpp"

#ifdef ENABLE_PAULIS
#include "operator/Operator.hpp"
#endif // ENABLE_PAULIS


namespace ann_on_gpu {

namespace kernel {

template<typename Basis_t>
struct Update_Policy;


#ifdef ENABLE_SPINS

template<>
struct Update_Policy<Spins> {
    template<typename Psi_t>
    HDINLINE void operator()(Spins& next_spins, const Spins& spins, const Psi_t& psi, void* rng_state) const {
        #include "cuda_kernel_defines.h"

        SINGLE {
            next_spins = spins.flip(random_uint32(rng_state) % psi.num_sites);
        }
    }

    Update_Policy& kernel() {
        return *this;
    }
};

#endif // ENABLE_SPINS


#ifdef ENABLE_PAULIS

template<>
struct Update_Policy<PauliString> {
    template<typename Psi_t>
    HDINLINE void operator()(PauliString& next_paulis, const PauliString& paulis, const Psi_t& psi, void* rng_state) const {
        #include "cuda_kernel_defines.h"

        SINGLE {
            const auto x = random_uint32(rng_state);

            next_paulis = paulis;
            next_paulis.set_at(x % psi.num_sites, x >> 30u);
        }
    }

    Update_Policy& kernel() {
        return *this;
    }
};

#endif // ENABLE_PAULIS

#ifdef ENABLE_FERMIONS

template<>
struct Update_Policy<Fermions> {
    template<typename Psi_t>
    HDINLINE void operator()(Fermions& next_fermions, const Fermions& fermions, const Psi_t& psi, void* rng_state) const {
        #include "cuda_kernel_defines.h"

        SINGLE {
            next_fermions = fermions.switch_at(random_uint32(rng_state) % psi.num_sites);
        }
    }

    Update_Policy& kernel() {
        return *this;
    }
};

#endif // ENABLE_FERMIONS

}  // namespace kernel


template<typename Basis_t>
struct Update_Policy;


#ifdef ENABLE_SPINS

template<>
struct Update_Policy<Spins> : public kernel::Update_Policy<Spins> {
    using kernel_t = kernel::Update_Policy<Spins>;
};

#endif // ENABLE_SPINS


#ifdef ENABLE_PAULIS

template<>
struct Update_Policy<PauliString> : public kernel::Update_Policy<PauliString> {
    using kernel_t = kernel::Update_Policy<PauliString>;
};

#endif // ENABLE_PAULIS


#ifdef ENABLE_FERMIONS

template<>
struct Update_Policy<Fermions> : public kernel::Update_Policy<Fermions> {
    using kernel_t = kernel::Update_Policy<Fermions>;
};

#endif // ENABLE_FERMIONS


}  // namespace ann_on_gpu
