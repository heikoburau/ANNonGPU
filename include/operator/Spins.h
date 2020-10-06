#pragma once

#include "MatrixElement.hpp"
#include "types.h"
#include "random.h"
#include <builtin_types.h>
#include <cstdint>
#include <random>
#include <array>

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"
#endif


namespace ann_on_gpu {


template<unsigned int num_types>
struct Spins_t;

namespace generic {

template<unsigned int num_types>
struct Spins_t {
    using type = uint64_t;
    // constexpr unsigned int num_types_half = num_types / 2u;

    type configurations[num_types];

    Spins_t() = default;

    HDINLINE Spins_t(
        const type configurations[num_types],
        const unsigned int num_spins
    ) {
        #ifdef __CUDA_ARCH__
        #pragma unroll
        #endif
        for(auto i = 0u; i < num_types; i++) {
            this->configurations[i] = configurations[i];
        }

        const auto type_idx = num_spins / 64u;
        if(type_idx < num_types) {
            this->configurations[type_idx] &= ((type)1u << (num_spins % 64)) - 1u;
        }
    }

    HDINLINE Spins_t(const Spins_t& other) {
        #ifdef __CUDA_ARCH__
        #pragma unroll
        #endif
        for(auto i = 0u; i < num_types; i++) {
            this->configurations[i] = other.configurations[i];
        }
    }

    HDINLINE Spins_t& operator=(const Spins_t& other) {
        #ifdef __CUDA_ARCH__
        #pragma unroll
        #endif
        for(auto i = 0u; i < num_types; i++) {
            this->configurations[i] = other.configurations[i];
        }

        return *this;
    }

    HDINLINE void set_randomly(void* random_state, const unsigned int num_spins) {
        #ifdef __CUDA_ARCH__
        #pragma unroll
        #endif
        for(auto i = 0u; i < num_types; i++) {
            this->configurations[i] = random_uint64(random_state);
        }

        const auto type_idx = num_spins / 64u;
        if(type_idx < num_types) {
            this->configurations[type_idx] &= ((type)1u << (num_spins % 64)) - 1u;
        }
    }

    HDINLINE float operator[](const unsigned int position) const {
        return 2.0f * static_cast<float>(
            static_cast<bool>(this->configurations[position / 64u] & ((type)1u << (position % 64u)))
        ) - 1.0f;
    }

    HDINLINE int network_unit_at(const unsigned int idx) const {
        // caution: this implementation has to be consistend with `PsiDeep::update_angles()`

        return (
            2 * static_cast<int>(static_cast<bool>(
                this->configurations[idx / 64u] & ((type)1u << (idx % 64u))
            )) - 1
        );
    }

    #ifdef __PYTHONCC__

    decltype(auto) array(const unsigned int num_spins) const {
        xt::pytensor<double, 1u> result(std::array<long int, 1u>({num_spins}));
        for(auto i = 0u; i < num_spins; i++) {
            result[i] = (*this)[i];
        }

        return result;
    }

    #endif // __CUDACC__

    HDINLINE void flip(const unsigned int position) {
        this->configurations[position / 64u] ^= (type)1u << (position % 64u);
    }

    HDINLINE bool operator==(const Spins_t& other) const {
        #ifdef __CUDA_ARCH__
        #pragma unroll
        #endif
        for(auto i = 0u; i < num_types; i++) {
            if(this->configurations[i] != other.configurations[i]) {
                return false;
            }
        }
        return true;
    }

    HDINLINE bool operator!=(const Spins_t& other) const {
        #ifdef __CUDA_ARCH__
        #pragma unroll
        #endif
        for(auto i = 0u; i < num_types; i++) {
            if(this->configurations[i] != other.configurations[i]) {
                return true;
            }
        }
        return false;
    }

    HDINLINE Spins_t& operator^=(const Spins_t& other) {
        #ifdef __CUDA_ARCH__
        #pragma unroll
        #endif
        for(auto i = 0u; i < num_types; i++) {
            this->configurations[i] ^= other.configurations[i];
        }

        return *this;
    }

    HDINLINE Spins_t& operator&=(const Spins_t& other) {
        #ifdef __CUDA_ARCH__
        #pragma unroll
        #endif
        for(auto i = 0u; i < num_types; i++) {
            this->configurations[i] &= other.configurations[i];
        }

        return *this;
    }

    HDINLINE Spins_t& operator|=(const Spins_t& other) {
        #ifdef __CUDA_ARCH__
        #pragma unroll
        #endif
        for(auto i = 0u; i < num_types; i++) {
            this->configurations[i] |= other.configurations[i];
        }

        return *this;
    }

    HDINLINE void rotate_left_by_one(const unsigned int num_spins) {

        #include "cuda_kernel_defines.h"
        SHARED bool little_bit, big_bit;

        little_bit = this->configurations[num_types - 1] & ((type)1u << ((num_spins - 1u) % 64u));

        #ifdef __CUDA_ARCH__
        #pragma unroll
        #endif
        for(auto i = 0; i < num_types; i++) {
            big_bit = this->configurations[i] & ((type)1u << 63u);

            this->configurations[i] <<= 1u;
            this->configurations[i] |= (type)little_bit;

            little_bit = big_bit;
        }

        this->configurations[num_types - 1] &= ~(1u << (num_spins % 64u));
    }

    HDINLINE int total_z(const unsigned int num_spins) const {
        auto result = 0;

        for(auto i = 0; i < num_types - 1u; i++) {
            #ifdef __CUDA_ARCH__
                result += 2 * bit_count(this->configurations[i]) - 64;
            #else
                result += 2 * bit_count(this->configurations[i]) - 64;
            #endif
        }
        const auto type_idx = num_spins / 64u;
        if(type_idx < num_types) {
            #ifdef __CUDA_ARCH__
                result += 2 * bit_count(this->configurations[num_types - 1u] & ((1u << (num_spins % 64u)) - 1)) - (num_spins % 64u);
            #else
                result += 2 * bit_count(this->configurations[num_types - 1u] & ((1u << (num_spins % 64u)) - 1)) - (num_spins % 64u);
            #endif
        }

        return result;
    }

    HDINLINE Spins_t<1u> extract_first_64() const {
        Spins_t<1u> result;
        result.configurations[0] = this->configurations[0];
        return result;
    }
};

}  // namespace generic


template<unsigned int num_types>
struct Spins_t : public generic::Spins_t<num_types> {
    Spins_t() = default;

    HDINLINE Spins_t(const Spins_t& other) {
        #ifdef __CUDA_ARCH__
        #pragma unroll
        #endif
        for(auto i = 0u; i < num_types; i++) {
            this->configurations[i] = other.configurations[i];
        }
    }

    HDINLINE Spins_t(const generic::Spins_t<num_types>& other) {
        #ifdef __CUDA_ARCH__
        #pragma unroll
        #endif
        for(auto i = 0u; i < num_types; i++) {
            this->configurations[i] = other.configurations[i];
        }
    }
};

template<>
struct Spins_t<1u> : public generic::Spins_t<1u> {

    Spins_t<1u>() = default;

    HDINLINE Spins_t<1u>(type configuration, const unsigned int num_spins) {
        if(num_spins == 64u) {
            this->configuration() = configuration;
        }
        else {
            this->configuration() = configuration & (((type)1u << num_spins) - 1u);
        }
    }

    HDINLINE Spins_t<1u>(generic::Spins_t<1u> other) {
        this->configuration() = other.configurations[0];
    }

    HDINLINE type& configuration() {
        return this->configurations[0];
    }

    HDINLINE const type& configuration() const {
        return this->configurations[0];
    }

    HDINLINE unsigned int hamming_distance(const Spins_t<1u>& other) const {
        return bit_count(this->configuration() ^ other.configuration());
    }

    HDINLINE uint64_t bit_at(const unsigned int i) const {
        return this->configuration() & ((type)1u << i);
    }

    HDINLINE Spins_t<1u> extract_first_n(const unsigned int n) const {
        return Spins_t<1u>(this->configuration(), n);
    }

    HDINLINE type is_different(const Spins_t<1u>& other) const {
        return this->configuration() ^ other.configuration();
    }

    HDINLINE Spins_t<1u>& operator=(const Spins_t<1u>& other) {
        this->configuration() = other.configuration();

        return *this;
    }

    HDINLINE Spins_t<1u>& operator=(const generic::Spins_t<1u>& other) {
        this->configuration() = other.configurations[0];

        return *this;
    }

    HDINLINE bool operator==(const Spins_t<1u>& other) const {
        return this->configuration() == other.configuration();
    }

    HDINLINE bool operator!=(const Spins_t<1u>& other) const {
        return this->configuration() != other.configuration();
    }

    HDINLINE void flip(const unsigned int position) {
        this->configuration() ^= (type)1u << (position % 64u);
    }

    // todo: fix for N = 64
    HDINLINE Spins_t<1u> rotate_left(const unsigned int shift, const unsigned int N) const {
        return Spins_t<1u>(
            (this->configuration() << shift) | (this->configuration() >> (N - shift)),
            N
        );
    }

    HDINLINE Spins_t<1u> shift_vertical(
        const unsigned int shift, const unsigned int nrows, const unsigned int ncols
    ) const {
        return Spins_t<1u>(
            (this->configuration() << (shift * ncols)) | (this->configuration() >> ((nrows - shift) * ncols)),
            nrows * ncols
        );
    }

    HDINLINE Spins_t<1u> select_left_columns(const unsigned int select, const unsigned int nrows, const unsigned int ncols) const {
        const auto row = ((1u << select) - 1u) << (ncols - select);
        type mask = 0u;
        for(auto i = 0u; i < nrows; i++) {
            mask |= row << (i * ncols);
        }
        return Spins_t<1u>(this->configuration() & mask, nrows * ncols);
    }

    HDINLINE Spins_t<1u> select_right_columns(const unsigned int select, const unsigned int nrows, const unsigned int ncols) const {
        const auto row = (1u << select) - 1u;
        type mask = 0u;
        for(auto i = 0u; i < nrows; i++) {
            mask |= row << (i * ncols);
        }
        return Spins_t<1u>(this->configuration() & mask, nrows * ncols);
    }

    HDINLINE Spins_t<1u> shift_horizontal(
        const unsigned int shift, const unsigned int nrows, const unsigned int ncols
    ) const {
        const auto tmp = this->rotate_left(shift, nrows * ncols);
        return Spins_t<1u>(
            (
                tmp.select_left_columns(nrows - shift, nrows, ncols).configuration() |
                tmp.select_right_columns(shift, nrows, ncols).shift_vertical(nrows - 1, nrows, ncols).configuration()
            ),
            nrows * ncols
        );
    }

    HDINLINE Spins_t<1u> shift_2d(
        const unsigned int shift_i, const unsigned int shift_j,
        const unsigned int nrows, const unsigned int ncols
    ) const {
        return this->shift_vertical(shift_i, nrows, ncols).shift_horizontal(shift_j, nrows, ncols);
    }
};


#if MAX_SPINS <= 64
using Spins = Spins_t<1u>;
#elif MAX_SPINS <= 128
using Spins = Spins_t<2u>;
#elif MAX_SPINS <= 192
using Spins = Spins_t<3u>;
#elif MAX_SPINS <= 256
using Spins = Spins_t<4u>;
#endif

}  // namespace ann_on_gpu
