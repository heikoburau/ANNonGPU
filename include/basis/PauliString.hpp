#pragma once

#include "bit_operations.hpp"
#include "operator/MatrixElement.hpp"
#include "Spins.h"
#include "types.h"

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"
#endif


namespace ann_on_gpu {



struct PauliString {
    using dtype = uint64_t;

    dtype a, b;

    PauliString() = default;
    HDINLINE PauliString(const dtype& a, const dtype& b) : a(a), b(b) {};

    static HDINLINE unsigned int num_configurations(const unsigned int num_sites) {
        return 1u << (2u * num_sites);
    }

    static HDINLINE PauliString enumerate(const unsigned int index) {
        return PauliString(
            detail::pick_bits_at_even_sites(index),
            detail::pick_bits_at_even_sites(index >> 1u)
        );
    }

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

    HDINLINE unsigned int operator[](const unsigned int idx) const {
        return (
            int(bool(this->a & (1lu << idx))) |
            (int(bool(this->b & (1lu << idx))) << 1lu)
        );
    }

#ifdef __PYTHONCC__
    decltype(auto) array(const unsigned int num_sites) const {
        xt::pytensor<unsigned int, 1u> result(std::array<long int, 1u>({num_sites}));
        for(auto i = 0u; i < num_sites; i++) {
            result[i] = (*this)[i];
        }

        return result;
    }
#endif // __PYTHONCC__

    HDINLINE int network_unit_at(const unsigned int idx) const {
        // caution: this implementation has to be consistend with `PsiDeep::update_angles()`

        return 2 * static_cast<int>(
            (*this)[idx / 3u] - 1u == (idx % 3u)
        ) - 1;
    }

    HDINLINE bool contains(const int index) const {
        return (*this)[index];
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

    HDINLINE bool operator==(const PauliString& other) const {
        return (this->a == other.a) && (this->b == other.b);
    }

    HDINLINE bool operator!=(const PauliString& other) const {
        return (this->a != other.a) || (this->b != other.b);
    }

    HDINLINE bool operator<(const PauliString& other) const {
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

    HDINLINE dtype is_different(const PauliString& other) const {
        return (this->a ^ other.a) | (this->b ^ other.b);
    }

    HDINLINE dtype is_sigma_x() const {
        return (this->a) & (~this->b);
    }

    HDINLINE dtype is_sigma_y() const {
        return (~this->a) & (this->b);
    }

    HDINLINE dtype is_sigma_z() const {
        return (this->a) & (this->b);
    }

    HDINLINE dtype is_diagonal_bitwise() const {
        return ~(this->a ^ this->b);
    }

    HDINLINE bool is_diagonal() const {
        return !(this->a ^ this->b);
    }

#ifdef ENABLE_SPINS
    HDINLINE bool is_diagonal_on_basis(Spins) const {
        return this->is_diagonal();
    }

    HDINLINE bool applies_a_prefactor(Spins) const {
        return this->is_sigma_y() | this->is_sigma_z();
    }
#endif // ENABLE_SPINS

#ifdef ENABLE_PAULIS
    HDINLINE bool is_diagonal_on_basis(PauliString) const {
        return !this->is_non_trivial();
    }

    HDINLINE bool applies_a_prefactor(PauliString) const {
        return this->is_non_trivial();
    }
#endif // ENABLE_PAULIS

    HDINLINE bool has_no_sigma_yz() const {
        return !(this->is_sigma_y() | this->is_sigma_z());
    }

    HDINLINE dtype epsilon_is_negative(const PauliString& other) const {
        return (
            (this->is_sigma_x() & other.is_sigma_z()) |
            (this->is_sigma_y() & other.is_sigma_x()) |
            (this->is_sigma_z() & other.is_sigma_y())
        );
    }

    HDINLINE bool commutes_with(const PauliString& other) const {
        return !(
            bit_count(
                this->is_non_trivial() & other.is_non_trivial() & this->is_different(other)
            ) & 1lu
        );
    }

    HDINLINE complex_t complex_prefactor() const {
        complex_t result = 1.0;

        const auto num_sigma_y = bit_count(this->is_sigma_y());

        // is there a factor i*i=-1 left?
        if((num_sigma_y & 3u) > 1lu) {
            result *= -1.0;
        }

        if(num_sigma_y & 1lu) {
            result *= complex_t(0.0, -1.0);
        }

        return result;
    }

    HDINLINE PauliString roll(const unsigned int shift, const unsigned int N) const {
        return PauliString(
            (
                (this->a << shift) | (this->a >> (N - shift))
            ) & ((1lu << N) - 1lu),
            (
                (this->b << shift) | (this->b >> (N - shift))
            ) & ((1lu << N) - 1lu)
        );
    }

    HDINLINE PauliString rotate_to_smallest(unsigned int length) const {
        const auto mask = (1lu << length) - 1lu;
        auto result = *this;
        auto x = *this;

        for(auto i = 0u; i < length; i++) {
            const auto a_shifted = (x.a << 1u) | (x.a >> (length - 1u));
            const auto b_shifted = (x.b << 1u) | (x.b >> (length - 1u));

            x.a = a_shifted & mask;
            x.b = b_shifted & mask;

            if(x < result) {
                result = x;
            }
        }

        return result;
    }

#ifdef ENABLE_SPINS
    HDINLINE MatrixElement<Spins> apply(const Spins& spins) const {
        complex_t factor = this->complex_prefactor();

        Spins result_spins(spins);

        if(bit_count((~spins.configuration()) & (this->is_sigma_z() | this->is_sigma_y())) & 1lu) {
            factor *= -1.0;
        }

        result_spins.configuration() ^= (~this->is_diagonal_bitwise());

        return MatrixElement<Spins>{factor, result_spins};
    }
#endif // ENABLE_SPINS

    HDINLINE MatrixElement<PauliString> apply(const PauliString& x) const {
        complex_t factor = 1.0;
        if(bit_count(this->epsilon_is_negative(x)) & 1lu) {
            factor *= -1.0;
        }

        const auto num_epsilon = bit_count(
            this->is_non_trivial() & x.is_non_trivial() & this->is_different(x)
        );

        // is there a factor i*i=-1 left?
        if((num_epsilon & 3u) > 1lu) {
            factor *= -1.0;
        }

        if(num_epsilon & 1lu) {
            factor *= complex_t(0.0, -1.0);
        }

        return MatrixElement<PauliString>{factor, PauliString(this->a ^ x.a, this->b ^ x.b)};
    }
};


HDINLINE MatrixElement<PauliString> operator*(const PauliString& a, const PauliString& b) {
    return a.apply(b);
}


HDINLINE MatrixElement<PauliString> commutator(const PauliString& a, const PauliString& b) {
    const auto num_epsilon = bit_count(
        a.is_non_trivial() & b.is_non_trivial() & a.is_different(b)
    );

    if(!(num_epsilon & 1lu)) {
        return MatrixElement<PauliString>{complex_t(0.0), PauliString()};
    }

    complex_t factor = complex_t(0.0, 2.0);
    if(bit_count(a.epsilon_is_negative(b)) & 1lu) {
        factor *= -1.0;
    }

    // is there a factor i*i=-1 left?
    if((num_epsilon & 3u) == 3u) {
        factor *= -1.0;
    }

    return MatrixElement<PauliString>{factor, PauliString(a.a ^ b.a, a.b ^ b.b)};
}


}  // namespace ann_on_gpu
