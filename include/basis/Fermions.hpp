#pragma once

#include "types.h"
#include "random.h"
#include <builtin_types.h>
#include <cstdint>


namespace ann_on_gpu {


struct Fermions {
    using dtype = uint64_t;

    dtype configuration;

    Fermions() = default;
    HDINLINE Fermions(const dtype& conf) : configuration(conf) {}

    static HDINLINE Fermions enumerate(const unsigned int index) {
        return Fermions(index);
    }

    static HDINLINE unsigned int num_configurations(const unsigned int num_sites) {
        return 1u << num_sites;
    }

    HDINLINE double operator[](const unsigned int position) const {
        return static_cast<double>(
            static_cast<bool>(this->configuration & (1lu << position))
        );
    }

    HDINLINE double network_unit_at(const unsigned int idx) const {
        // caution: this implementation has to be consistent with `PsiDeep::update_angles()`

        return (*this)[idx];
    }

    HDINLINE bool operator==(const Fermions& other) const {
        return this->configuration == other.configuration;
    }

    HDINLINE bool operator!=(const Fermions& other) const {
        return this->configuration != other.configuration;
    }

    HDINLINE dtype is_different(const Fermions& other) const {
        return this->configuration ^ other.configuration;
    }

    HDINLINE void set_randomly(void* random_state, const unsigned int num_fermions) {
        this->configuration = random_uint64(random_state) & ((1lu << num_fermions) - 1lu);
    }

    HDINLINE Fermions switch_at(const unsigned int idx) const {
        Fermions result = *this;
        result.configuration ^= 1lu << idx;
        return result;
    }
};

}  // namespace ann_on_gpu
