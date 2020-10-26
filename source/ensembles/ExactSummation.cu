#ifdef ENABLE_EXACT_SUMMATION

#include "ensembles/ExactSummation.hpp"
#include "quantum_states.hpp"
#include "types.h"

#include <cassert>
#include <vector>
#include <algorithm>

using namespace std;


namespace ann_on_gpu {

template<typename Basis_t>
ExactSummation_t<Basis_t>::ExactSummation_t(const unsigned int num_sites, const bool gpu)
    :
    gpu(gpu)
        // allowed_spin_configurations_vec(nullptr)
    {
        this->num_sites = num_sites;
        this->num_configurations = Basis_t::num_configurations(num_sites);
        // this->has_total_z_symmetry = false;
    }

// void ExactSummation_t::set_total_z_symmetry(const int sector) {
//     vector<Spins> spins_tmp;
//     const auto hilbert_space_dim = pow(2, this->num_sites);

//     for(auto spin_index = 0u; spin_index < hilbert_space_dim; spin_index++) {
//         Spins spins(spin_index, this->num_sites);

//         if(spins.total_z(this->num_sites) == sector) {
//             spins_tmp.push_back(spins);
//         }
//     }

//     this->num_configurations = spins_tmp.size();
//     this->allowed_spin_configurations_vec = unique_ptr<Array<Spins>>(new Array<Spins>(spins_tmp.size(), this->gpu));
//     this->allowed_spin_configurations_vec->assign(spins_tmp.begin(), spins_tmp.end());
//     this->allowed_spin_configurations_vec->update_device();
//     this->allowed_spin_configurations = this->allowed_spin_configurations_vec->data();

//     this->has_total_z_symmetry = true;
// }


#ifdef ENABLE_SPINS
template struct ExactSummation_t<Spins>;
#endif // ENABLE_SPINS

#ifdef ENABLE_PAULIS
template struct ExactSummation_t<PauliString>;
#endif // ENABLE_PAULIS

} // namespace ann_on_gpu


#endif // ENABLE_EXACT_SUMMATION
