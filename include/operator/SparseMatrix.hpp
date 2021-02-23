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
    unsigned int   col_to_row[16];


    HDINLINE void apply(MatrixElement<PauliString>& x) {
        auto col = x.vector[this->site_i];

        if(this->two_sites) {
            col += 4u * x.vector[this->site_j];
        }
        const auto row = this->col_to_row[col];

        x.vector.set_at(this->site_i, row % 4u);
        if(this->two_sites) {
            x.vector.set_at(this->site_j, row / 4u);
        }

        x.coefficient = this->values[col];
    }

#ifdef __PYTHONCC__

    SparseMatrix(
        bool two_sites,
        unsigned site_i,
        unsigned site_j,
        const vector<double>& values,
        const vector<unsigned int>& col_to_row
    ) :
    two_sites(two_sites),
    site_i(site_i),
    site_j(site_j) {
        copy(values.begin(), values.end(), this->values);
        copy(col_to_row.begin(), col_to_row.end(), this->col_to_row);
    }

#endif // __PYTHONCC__
};

}  // namespace ann_on_gpu
