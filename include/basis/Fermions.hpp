#pragma once

#include "types.h"
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
        return 2.0 * static_cast<double>(
            static_cast<bool>(this->configuration & (1lu << position))
        ) - 1.0;
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
};

}  // namespace ann_on_gpu
