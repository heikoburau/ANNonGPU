#pragma once

#include "operator/MatrixElement.hpp"
#include "Spins.h"
#include "types.h"

#ifdef __CUDACC__
    namespace quantum_expression {
        struct FermionString;
    }
#else
    #include "QuantumExpression/QuantumExpression.hpp"
#endif


namespace ann_on_gpu {

using ::quantum_expression::FermionString;


struct FermionicString {
    using dtype = uint64_t;


    // encoding:

    // ba
    // 00 -> 1
    // 01 -> creation
    // 10 -> annihilation
    // 11 -> number
    dtype a, b;

    dtype bit_count_mask;

    FermionicString() = default;
    HDINLINE FermionicString(const dtype& a, const dtype& b, const dtype& bit_count_mask):
        a(a), b(b), bit_count_mask(bit_count_mask) {};

#ifndef __CUDACC__
    FermionicString(const FermionString& expr) : a(0lu), b(0lu), bit_count_mask(0lu) {
        for(auto symbol_it = expr.rbegin(); symbol_it != expr.rend(); symbol_it++) {
            const auto symbol = *symbol_it;

            this->set_at(symbol.index, symbol.op.type);

            if((symbol.op.type == 1u) || (symbol.op.type == 2u)) {
                this->bit_count_mask ^= (1lu << symbol.index) - 1lu;
            }
        }
    }
#endif // __CUDACC__

    HDINLINE void set_at(const unsigned int idx, const unsigned int type) {
        switch(type)
        {
        case 0u:
            this->a &= ~(1lu << idx);
            this->b &= ~(1lu << idx);
            break;
        case 1lu:
            this->a |= 1lu << idx;
            this->b &= ~(1lu << idx);
            break;
        case 2u:
            this->a &= ~(1lu << idx);
            this->b |= 1lu << idx;
            break;
        case 3u:
            this->a |= 1lu << idx;
            this->b |= 1lu << idx;
        }
    }

    HDINLINE size_t size() const {
        return bit_count(this->a | this->b);
    }

    HDINLINE operator bool() const {
        return this->size();
    }

    HDINLINE bool cast_to_bool() const {
        return *this;
    }

    HDINLINE bool operator==(const FermionicString& other) const {
        return (this->a == other.a) && (this->b == other.b);
    }

    HDINLINE bool operator!=(const FermionicString& other) const {
        return (this->a != other.a) || (this->b != other.b);
    }

    HDINLINE bool operator<(const FermionicString& other) const {
        if(this->a == other.a) {
            return this->b < other.b;
        }
        else {
            return this->a < other.a;
        }
    }

    HDINLINE dtype is_non_trivial() const {
        return this->a | this->b;
    }

    HDINLINE dtype is_creation() const {
        return this->a & (~this->b);
    }

    HDINLINE dtype is_annihiliation() const {
        return (~this->a) & this->b;
    }

    HDINLINE dtype is_number() const {
        return this->a & this->b;
    }

    HDINLINE dtype is_annihilation_or_number() const {
        return this->b;
    }

    HDINLINE dtype is_different(const FermionicString& other) const {
        return (this->a ^ other.a) | (this->b ^ other.b);
    }

    HDINLINE dtype is_diagonal_bitwise() const {
        return ~(this->a ^ this->b);
    }

    HDINLINE bool is_diagonal() const {
        return !(this->a ^ this->b);
    }

#ifdef ENABLE_FERMIONS
    HDINLINE MatrixElement<Fermions> apply(const Fermions& fermions) const {
        const auto conf = fermions.configuration;

        if(
            (this->is_creation() & conf) ||
            (this->is_annihilation_or_number() & (~conf))
        ) {
            return MatrixElement<Fermions>{complex_t(0.0), fermions};
        }

        auto result_conf = fermions.configuration;

        result_conf |= this->is_creation();
        result_conf &= ~this->is_annihiliation();

        return MatrixElement<Fermions>{
            (bit_count(conf & this->bit_count_mask) & 1u) ? complex_t(-1.0) : complex_t(1.0),
            Fermions(result_conf)
        };
    }
#endif // ENABLE_FERMIONS

};


}  // namespace ann_on_gpu
