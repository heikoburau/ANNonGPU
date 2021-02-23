#pragma once

#include "operator/MatrixElement.hpp"
#include "basis/PauliString.hpp"
// #include "tiny_complex.hpp"
#include <vector>
#include <algorithm>
#include "types.h"


namespace ann_on_gpu {


struct SparseMatrix {
    bool           two_sites;
    unsigned int   site_i, site_j;
    double         values[16];
    unsigned int   row_to_col[16];


    HDINLINE void apply(MatrixElement<PauliString>& x) {
        auto pauli_type = x.vector[this->site_i];

        if(this->two_sites) {
            pauli_type += 4u * x.vector[this->site_j];
        }
        const auto col = this->row_to_col[pauli_type];

        x.vector.set_at(this->site_i, col % 4u);
        if(this->two_sites) {
            x.vector.set_at(this->site_j, col / 4u);
        }

        x.coefficient = this->values[pauli_type];
    }

#ifdef __PYTHONCC__

    SparseMatrix(
        bool two_sites,
        unsigned site_i,
        unsigned site_j,
        const vector<double>& values,
        const vector<unsigned int>& row_to_col
    ) :
    two_sites(two_sites),
    site_i(site_i),
    site_j(site_j) {
        copy(values.begin(), values.end(), this->values);
        copy(row_to_col.begin(), row_to_col.end(), this->row_to_col);
    }

#endif // __PYTHONCC__
};

}  // namespace ann_on_gpu
