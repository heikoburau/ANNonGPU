#include "quantum_state/PsiDeepSigned.hpp"
#include "ensembles.hpp"

#include <complex>
#include <vector>
#include <random>
#include <cstring>
#include <algorithm>
#include <iterator>


namespace ann_on_gpu {

using namespace cuda_complex;

template<bool symmetric>
PsiDeepSigned_t<symmetric>::PsiDeepSigned_t(const PsiDeepSigned_t& other)
    :
    psi_plus(other.psi_plus),
    psi_minus(other.psi_minus),
    gpu(other.gpu)
{
    this->N = other.N;
    this->num_sites = other.num_sites;

    this->num_params = other.num_params;

    this->prefactor = other.prefactor;
    this->log_prefactor = other.log_prefactor;

    this->init_kernel();
}


template<bool symmetric>
PsiDeepSigned_t<symmetric>& PsiDeepSigned_t<symmetric>::operator=(const PsiDeepSigned_t& other) {
    this->psi_plus = other.psi_plus;
    this->psi_minus = other.psi_minus;
    this->gpu = other.gpu;

    this->N = other.N;
    this->num_sites = other.num_sites;

    this->num_params = other.num_params;

    this->prefactor = other.prefactor;
    this->log_prefactor = other.log_prefactor;

    this->init_kernel();

    return *this;
}

template<bool symmetric>
void PsiDeepSigned_t<symmetric>::init_kernel() {
    this->kernel().psi_plus = this->psi_plus.kernel();
    this->kernel().psi_minus = this->psi_minus.kernel();

    this->update_kernel();
}


template<bool symmetric>
void PsiDeepSigned_t<symmetric>::update_kernel() {
    this->psi_plus.update_kernel();
    this->psi_minus.update_kernel();
}

template<bool symmetric>
Array<double> PsiDeepSigned_t<symmetric>::get_params() const {
    Array<double> result(this->num_params, false);

    const auto params_plus = this->psi_plus.get_params();
    const auto params_minus = this->psi_minus.get_params();

    copy(params_plus.begin(), params_plus.end(), result.begin());
    copy(params_minus.begin(), params_minus.end(), result.begin() + this->psi_plus.num_params);

    return result;
}


template<bool symmetric>
void PsiDeepSigned_t<symmetric>::set_params(const Array<double>& new_params) {

    this->psi_plus.set_params(Array<double>(
        std::vector<double>(new_params.begin(), new_params.begin() + this->psi_plus.num_params),
        false
    ));

    this->psi_minus.set_params(Array<double>(
        std::vector<double>(new_params.begin() + this->psi_plus.num_params, new_params.end()),
        false
    ));
}


#ifdef PSI_DEEP_SYMMETRIC
template struct PsiDeepSigned_t<true>;
#else
template struct PsiDeepSigned_t<false>;
#endif // PSI_DEEP_SYMMETRIC

} // namespace ann_on_gpu
