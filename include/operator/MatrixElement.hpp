#pragma once

#include "types.h"


namespace ann_on_gpu {

template<typename Basis_t>
struct MatrixElement {
    MatrixElement() = default;

    complex_t   coefficient;
    Basis_t     vector;
};

}  // namespace ann_on_gpu
