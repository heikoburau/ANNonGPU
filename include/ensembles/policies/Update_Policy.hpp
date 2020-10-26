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
            next_spins = spins.flip(random_uint32(rng_state) % psi.get_num_input_units());
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
    Operator op;


    template<typename Psi_t>
    HDINLINE void operator()(PauliString& next_paulis, const PauliString& paulis, const Psi_t& psi, void* rng_state) const {
        #include "cuda_kernel_defines.h"

        SINGLE {
            next_paulis = paulis.apply(
                this->op.pauli_strings[
                    random_uint32(rng_state) % this->op.num_strings
                ]
            ).vector;
        }
    }

    Update_Policy& kernel() {
        return *this;
    }
};

#endif // ENABLE_PAULIS

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

    Operator op;

    inline Update_Policy(const Operator& op) : op(op) {
        this->kernel().op = this->op.kernel();
    }

    inline Update_Policy(const Update_Policy& other) : op(other.op) {
        this->kernel().op = this->op.kernel();
    }
};

#endif // ENABLE_PAULIS


}  // namespace ann_on_gpu
