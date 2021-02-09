#pragma once

#include "operator/MatrixElement.hpp"
#include "basis/PauliString.hpp"
// #include "tiny_complex.hpp"
#include "types.h"


namespace ann_on_gpu {


struct Matrix4x4 {
    unsigned int   site;
    double         values[4];
    char           pauli_types[4];
    bool           is_diagonal;


    HDINLINE void apply(MatrixElement<PauliString>& x) {
        const auto pauli_type = x.vector[this->site];

        if(!this->is_diagonal) {
            x.vector.set_at(this->site, this->pauli_types[pauli_type]);
        }

        x.coefficient *= this->values[pauli_type];
    }
};


}  // namespace ann_on_gpu
