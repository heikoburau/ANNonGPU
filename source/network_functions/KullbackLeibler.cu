// ***********************************************************
// *       This is an automatically generated file.          *
// *       For editing, please use the source file:          *
// KullbackLeibler.cu.template
// ***********************************************************

#ifndef LEAN_AND_MEAN

#include "network_functions/KullbackLeibler.hpp"
#include "ensembles.hpp"
#include "quantum_states.hpp"

#include <cstring>
#include <math.h>


namespace ann_on_gpu {

namespace kernel {


template<bool compute_gradient, bool noise, typename Psi_t, typename PsiPrime_t, typename Ensemble>
void kernel::KullbackLeibler::compute_averages(
    Psi_t& psi, PsiPrime_t& psi_prime, Ensemble& ensemble, double threshold
) const {
    const auto this_ = *this;
    const auto psi_kernel = psi.kernel();
    const auto psi_prime_kernel = psi_prime.kernel();
    const auto threshold2 = threshold * threshold;

    ensemble.foreach(
        psi_prime,
        [=] __device__ __host__ (
            const unsigned int spin_index,
            const typename Ensemble::Basis_t& configuration,
            const complex_t& log_psi_prime,
            typename PsiPrime_t::Payload& payload_prime,
            const double& weight_prime
        ) {
            #include "cuda_kernel_defines.h"

            SHARED typename Psi_t::dtype    log_psi;
            SHARED typename Psi_t::Payload  payload;
            SHARED complex_t                deviation;
            SHARED double                   deviation2;
            SHARED double                   prob_ratio;
            SHARED double                   weight;

            psi_kernel.init_payload(payload, configuration, spin_index);
            psi_kernel.log_psi_s(log_psi, configuration, payload);

            SINGLE {
                log_psi *= this_.log_psi_scale;

                prob_ratio = exp(2.0 * (log_psi.real() - log_psi_prime.real()));
                weight = prob_ratio * weight_prime;
                generic_atomicAdd(this_.total_weight, weight);
                generic_atomicAdd(this_.mean_deviation, weight * (log_psi_prime - log_psi));
                deviation = log_psi_prime - log_psi - *this_.last_mean_deviation;

                deviation2 = abs2(deviation);
                if(deviation2 > threshold2) {
                    generic_atomicAdd(this_.deviation, weight * deviation);
                    generic_atomicAdd(this_.deviation2, weight * deviation2);
                }
            }
            SYNC;

            if(compute_gradient) {
                psi_prime_kernel.foreach_O_k(
                    configuration,
                    payload_prime,
                    [&](const unsigned int k, const complex_t& O_k) {
                        generic_atomicAdd(
                            &this_.O_k[k],
                            weight * O_k
                        );
                        if(deviation2 > threshold2) {
                            generic_atomicAdd(
                                &this_.deviation_O_k_conj[k],
                                weight * deviation * conj(O_k)
                            );
                        }

                        if(noise) {
                            const auto O_k2 = abs2(O_k);

                            if(deviation2 > threshold2) {
                                generic_atomicAdd(
                                    &this_.deviation2_O_k2[k],
                                    weight * deviation2 * O_k2
                                );
                                generic_atomicAdd(
                                    &this_.deviation_O_k[k],
                                    weight * deviation * O_k
                                );
                                generic_atomicAdd(
                                    &this_.deviation_O_k2[k],
                                    weight * deviation * O_k2
                                );
                                generic_atomicAdd(
                                    &this_.deviation2_O_k[k],
                                    weight * deviation2 * O_k
                                );

                            }
                            generic_atomicAdd(
                                &this_.O_k2[k],
                                weight * O_k2
                            );
                        }
                    }
                );
            }
        },
        max(psi.get_width(), psi_prime.get_width())
    );
}

} // namespace kernel

KullbackLeibler::KullbackLeibler(const unsigned int num_params, const bool gpu)
      : num_params(num_params),
        deviation(1, gpu),
        deviation2(1, gpu),
        O_k(num_params, gpu),
        deviation_O_k_conj(num_params, gpu),
        deviation2_O_k2(num_params, gpu),
        deviation_O_k(num_params, gpu),
        deviation2_O_k(num_params, gpu),
        deviation_O_k2(num_params, gpu),
        O_k2(num_params, gpu),
        mean_deviation(1, gpu),
        last_mean_deviation(1, gpu),
        total_weight(1, gpu)
    {
    this->gpu = gpu;

    this->kernel().deviation = this->deviation.data();
    this->kernel().deviation2 = this->deviation2.data();
    this->kernel().O_k = this->O_k.data();
    this->kernel().deviation_O_k_conj = this->deviation_O_k_conj.data();

    this->kernel().deviation2_O_k2 = this->deviation2_O_k2.data();
    this->kernel().deviation_O_k = this->deviation_O_k.data();
    this->kernel().deviation2_O_k = this->deviation2_O_k.data();
    this->kernel().deviation_O_k2 = this->deviation_O_k2.data();
    this->kernel().O_k2 = this->O_k2.data();

    this->kernel().mean_deviation = this->mean_deviation.data();
    this->kernel().last_mean_deviation = this->last_mean_deviation.data();

    this->kernel().total_weight = this->total_weight.data();

    this->last_mean_deviation.clear();

    this->log_psi_scale = 1.0;
}


void KullbackLeibler::clear() {
    this->deviation.clear();
    this->deviation2.clear();
    this->O_k.clear();
    this->deviation_O_k_conj.clear();

    this->deviation2_O_k2.clear();
    this->deviation_O_k.clear();
    this->deviation2_O_k.clear();
    this->deviation_O_k2.clear();
    this->O_k2.clear();

    this->mean_deviation.clear();

    this->total_weight.clear();
}

void KullbackLeibler::update_last_mean_deviation() {
    this->mean_deviation.update_host();
    this->mean_deviation.front() /= this->total_weight.front();

    this->last_mean_deviation.front() = this->mean_deviation.front();
    this->last_mean_deviation.update_device();
}

template<typename Psi_t, typename PsiPrime_t, typename Ensemble>
double KullbackLeibler::value(
    Psi_t& psi, PsiPrime_t& psi_prime, Ensemble& ensemble, double threshold
) {
    this->clear();
    this->compute_averages<false, false>(psi, psi_prime, ensemble, threshold);
    this->total_weight.update_host();
    this->update_last_mean_deviation();

    this->deviation.update_host();
    this->deviation2.update_host();

    this->deviation.front() /= this->total_weight.front();
    this->deviation2.front() /= this->total_weight.front();

    return sqrt(max(
        1e-8,
        this->deviation2.front() - abs2(this->deviation.front())
    ));

    // return this->deviation2.front() - abs2(this->deviation.front());
}


template<typename Psi_t, typename PsiPrime_t, typename Ensemble>
double KullbackLeibler::gradient(
    complex<double>* result, Psi_t& psi, PsiPrime_t& psi_prime, Ensemble& ensemble, const double nu, double threshold
) {
    this->clear();
    this->compute_averages<true, false>(psi, psi_prime, ensemble, threshold);
    this->total_weight.update_host();
    this->update_last_mean_deviation();

    this->deviation.update_host();
    this->deviation2.update_host();
    this->O_k.update_host();
    this->deviation_O_k_conj.update_host();


    this->deviation.front() /= this->total_weight.front();
    this->deviation2.front() /= this->total_weight.front();
    for(auto k = 0u; k < this->num_params; k++) {
        this->O_k[k] /= this->total_weight.front();
        this->deviation_O_k_conj[k] /= this->total_weight.front();
    }


    const auto value = sqrt(max(
        1e-8,
        this->deviation2.front() - abs2(this->deviation.front())
    ));
    const auto factor = pow(value, nu);

    for(auto k = 0u; k < this->num_params; k++) {
        result[k] = (
            this->deviation_O_k_conj[k] - this->deviation.front() * conj(this->O_k[k])
        ).to_std() / factor;
    }

    return value;
}

template<typename Psi_t, typename Psi_t_prime, typename Ensemble>
tuple<
    Array<complex_t>,
    Array<double>,
    double
> KullbackLeibler::gradient_with_noise(
    Psi_t& psi, Psi_t_prime& psi_prime, Ensemble& ensemble, const double nu, double threshold
) {
    this->clear();
    this->compute_averages<true, true>(psi, psi_prime, ensemble, threshold);
    this->total_weight.update_host();
    this->update_last_mean_deviation();

    this->deviation.update_host();
    this->deviation2.update_host();
    this->O_k.update_host();
    this->deviation_O_k_conj.update_host();

    this->deviation2_O_k2.update_host();
    this->deviation_O_k.update_host();
    this->deviation_O_k2.update_host();
    this->deviation2_O_k.update_host();
    this->O_k2.update_host();

    this->deviation.front() /= this->total_weight.front();
    this->deviation2.front() /= this->total_weight.front();
    for(auto k = 0u; k < this->num_params; k++) {
        this->O_k[k] /= this->total_weight.front();
        this->deviation_O_k_conj[k] /= this->total_weight.front();

        this->deviation2_O_k2[k] /= this->total_weight.front();
        this->deviation_O_k[k] /= this->total_weight.front();
        this->deviation_O_k2[k] /= this->total_weight.front();
        this->deviation2_O_k[k] /= this->total_weight.front();
        this->O_k2[k] /= this->total_weight.front();
    }


    const auto value = sqrt(max(
        1e-8,
        this->deviation2.front() - abs2(this->deviation.front())
    ));
    const auto factor = pow(value, nu);

    Array<complex_t> result(this->num_params, false);
    Array<double> noise(this->num_params, false);

    const auto d = this->deviation.front();
    const auto d2 = this->deviation2.front();

    for(auto k = 0u; k < this->num_params; k++) {
        const auto O_k = this->O_k[k];
        const auto d_O_k_conj = this->deviation_O_k_conj[k];

        const auto d2_O_k2 = this->deviation2_O_k2[k];
        const auto d_O_k = this->deviation_O_k[k];
        const auto d_O_k2 = this->deviation_O_k2[k];
        const auto d2_O_k = this->deviation2_O_k[k];
        const auto O_k2 = this->O_k2[k];

        result[k] = (
            d_O_k_conj - d * conj(O_k)
        ) / factor;

        noise[k] = sqrt(
            (
                d2_O_k2 - abs2(d_O_k_conj) + 2.0 * (
                    d_O_k * conj(d) * conj(O_k) + 2.0 * conj(d_O_k_conj) * d * conj(O_k)
                    -d2_O_k * conj(O_k) - d_O_k2 * conj(d)
                ).real() + d2 * abs2(O_k) + abs2(d) * O_k2 - 4.0 * abs2(d) * abs2(O_k)
            ) / ensemble.get_num_steps()
        ) / factor;
    }

    return make_tuple(move(result), move(noise), value);
}


#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_DEEP) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(PsiClassicalFP<1u>&, PsiDeep&, MonteCarlo_tt<Spins>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalFP<1u>&, PsiDeep&, MonteCarlo_tt<Spins>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalFP<1u>&, PsiDeep&, MonteCarlo_tt<Spins>&, const double, double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_DEEP) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(PsiClassicalFP<2u>&, PsiDeep&, MonteCarlo_tt<Spins>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalFP<2u>&, PsiDeep&, MonteCarlo_tt<Spins>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalFP<2u>&, PsiDeep&, MonteCarlo_tt<Spins>&, const double, double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_DEEP) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(PsiClassicalANN<1u>&, PsiDeep&, MonteCarlo_tt<Spins>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalANN<1u>&, PsiDeep&, MonteCarlo_tt<Spins>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalANN<1u>&, PsiDeep&, MonteCarlo_tt<Spins>&, const double, double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_DEEP) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(PsiClassicalANN<2u>&, PsiDeep&, MonteCarlo_tt<Spins>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalANN<2u>&, PsiDeep&, MonteCarlo_tt<Spins>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalANN<2u>&, PsiDeep&, MonteCarlo_tt<Spins>&, const double, double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CNN) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(PsiClassicalFP<1u>&, PsiCNN&, MonteCarlo_tt<Spins>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalFP<1u>&, PsiCNN&, MonteCarlo_tt<Spins>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalFP<1u>&, PsiCNN&, MonteCarlo_tt<Spins>&, const double, double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CNN) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(PsiClassicalFP<2u>&, PsiCNN&, MonteCarlo_tt<Spins>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalFP<2u>&, PsiCNN&, MonteCarlo_tt<Spins>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalFP<2u>&, PsiCNN&, MonteCarlo_tt<Spins>&, const double, double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CNN) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(PsiClassicalANN<1u>&, PsiCNN&, MonteCarlo_tt<Spins>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalANN<1u>&, PsiCNN&, MonteCarlo_tt<Spins>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalANN<1u>&, PsiCNN&, MonteCarlo_tt<Spins>&, const double, double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CNN) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(PsiClassicalANN<2u>&, PsiCNN&, MonteCarlo_tt<Spins>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalANN<2u>&, PsiCNN&, MonteCarlo_tt<Spins>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalANN<2u>&, PsiCNN&, MonteCarlo_tt<Spins>&, const double, double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_DEEP) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(PsiClassicalFP<1u>&, PsiDeep&, MonteCarlo_tt<PauliString>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalFP<1u>&, PsiDeep&, MonteCarlo_tt<PauliString>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalFP<1u>&, PsiDeep&, MonteCarlo_tt<PauliString>&, const double, double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_DEEP) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(PsiClassicalFP<2u>&, PsiDeep&, MonteCarlo_tt<PauliString>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalFP<2u>&, PsiDeep&, MonteCarlo_tt<PauliString>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalFP<2u>&, PsiDeep&, MonteCarlo_tt<PauliString>&, const double, double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_DEEP) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(PsiClassicalANN<1u>&, PsiDeep&, MonteCarlo_tt<PauliString>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalANN<1u>&, PsiDeep&, MonteCarlo_tt<PauliString>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalANN<1u>&, PsiDeep&, MonteCarlo_tt<PauliString>&, const double, double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_DEEP) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(PsiClassicalANN<2u>&, PsiDeep&, MonteCarlo_tt<PauliString>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalANN<2u>&, PsiDeep&, MonteCarlo_tt<PauliString>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalANN<2u>&, PsiDeep&, MonteCarlo_tt<PauliString>&, const double, double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CNN) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(PsiClassicalFP<1u>&, PsiCNN&, MonteCarlo_tt<PauliString>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalFP<1u>&, PsiCNN&, MonteCarlo_tt<PauliString>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalFP<1u>&, PsiCNN&, MonteCarlo_tt<PauliString>&, const double, double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CNN) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(PsiClassicalFP<2u>&, PsiCNN&, MonteCarlo_tt<PauliString>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalFP<2u>&, PsiCNN&, MonteCarlo_tt<PauliString>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalFP<2u>&, PsiCNN&, MonteCarlo_tt<PauliString>&, const double, double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CNN) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(PsiClassicalANN<1u>&, PsiCNN&, MonteCarlo_tt<PauliString>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalANN<1u>&, PsiCNN&, MonteCarlo_tt<PauliString>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalANN<1u>&, PsiCNN&, MonteCarlo_tt<PauliString>&, const double, double);
#endif
#if defined(ENABLE_MONTE_CARLO) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CNN) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(PsiClassicalANN<2u>&, PsiCNN&, MonteCarlo_tt<PauliString>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalANN<2u>&, PsiCNN&, MonteCarlo_tt<PauliString>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalANN<2u>&, PsiCNN&, MonteCarlo_tt<PauliString>&, const double, double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_DEEP) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(PsiClassicalFP<1u>&, PsiDeep&, ExactSummation_t<Spins>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalFP<1u>&, PsiDeep&, ExactSummation_t<Spins>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalFP<1u>&, PsiDeep&, ExactSummation_t<Spins>&, const double, double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_DEEP) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(PsiClassicalFP<2u>&, PsiDeep&, ExactSummation_t<Spins>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalFP<2u>&, PsiDeep&, ExactSummation_t<Spins>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalFP<2u>&, PsiDeep&, ExactSummation_t<Spins>&, const double, double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_DEEP) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(PsiClassicalANN<1u>&, PsiDeep&, ExactSummation_t<Spins>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalANN<1u>&, PsiDeep&, ExactSummation_t<Spins>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalANN<1u>&, PsiDeep&, ExactSummation_t<Spins>&, const double, double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_DEEP) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(PsiClassicalANN<2u>&, PsiDeep&, ExactSummation_t<Spins>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalANN<2u>&, PsiDeep&, ExactSummation_t<Spins>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalANN<2u>&, PsiDeep&, ExactSummation_t<Spins>&, const double, double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CNN) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(PsiClassicalFP<1u>&, PsiCNN&, ExactSummation_t<Spins>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalFP<1u>&, PsiCNN&, ExactSummation_t<Spins>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalFP<1u>&, PsiCNN&, ExactSummation_t<Spins>&, const double, double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CNN) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(PsiClassicalFP<2u>&, PsiCNN&, ExactSummation_t<Spins>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalFP<2u>&, PsiCNN&, ExactSummation_t<Spins>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalFP<2u>&, PsiCNN&, ExactSummation_t<Spins>&, const double, double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CNN) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(PsiClassicalANN<1u>&, PsiCNN&, ExactSummation_t<Spins>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalANN<1u>&, PsiCNN&, ExactSummation_t<Spins>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalANN<1u>&, PsiCNN&, ExactSummation_t<Spins>&, const double, double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_SPINS) && defined(ENABLE_PSI_CNN) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(PsiClassicalANN<2u>&, PsiCNN&, ExactSummation_t<Spins>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalANN<2u>&, PsiCNN&, ExactSummation_t<Spins>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalANN<2u>&, PsiCNN&, ExactSummation_t<Spins>&, const double, double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_DEEP) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(PsiClassicalFP<1u>&, PsiDeep&, ExactSummation_t<PauliString>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalFP<1u>&, PsiDeep&, ExactSummation_t<PauliString>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalFP<1u>&, PsiDeep&, ExactSummation_t<PauliString>&, const double, double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_DEEP) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(PsiClassicalFP<2u>&, PsiDeep&, ExactSummation_t<PauliString>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalFP<2u>&, PsiDeep&, ExactSummation_t<PauliString>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalFP<2u>&, PsiDeep&, ExactSummation_t<PauliString>&, const double, double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_DEEP) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(PsiClassicalANN<1u>&, PsiDeep&, ExactSummation_t<PauliString>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalANN<1u>&, PsiDeep&, ExactSummation_t<PauliString>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalANN<1u>&, PsiDeep&, ExactSummation_t<PauliString>&, const double, double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_DEEP) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(PsiClassicalANN<2u>&, PsiDeep&, ExactSummation_t<PauliString>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalANN<2u>&, PsiDeep&, ExactSummation_t<PauliString>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalANN<2u>&, PsiDeep&, ExactSummation_t<PauliString>&, const double, double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CNN) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(PsiClassicalFP<1u>&, PsiCNN&, ExactSummation_t<PauliString>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalFP<1u>&, PsiCNN&, ExactSummation_t<PauliString>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalFP<1u>&, PsiCNN&, ExactSummation_t<PauliString>&, const double, double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CNN) && defined(ENABLE_PSI_CLASSICAL)
template double KullbackLeibler::value(PsiClassicalFP<2u>&, PsiCNN&, ExactSummation_t<PauliString>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalFP<2u>&, PsiCNN&, ExactSummation_t<PauliString>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalFP<2u>&, PsiCNN&, ExactSummation_t<PauliString>&, const double, double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CNN) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(PsiClassicalANN<1u>&, PsiCNN&, ExactSummation_t<PauliString>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalANN<1u>&, PsiCNN&, ExactSummation_t<PauliString>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalANN<1u>&, PsiCNN&, ExactSummation_t<PauliString>&, const double, double);
#endif
#if defined(ENABLE_EXACT_SUMMATION) && defined(ENABLE_PAULIS) && defined(ENABLE_PSI_CNN) && defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_CLASSICAL_ANN)
template double KullbackLeibler::value(PsiClassicalANN<2u>&, PsiCNN&, ExactSummation_t<PauliString>&, double);
template double KullbackLeibler::gradient(complex<double>*, PsiClassicalANN<2u>&, PsiCNN&, ExactSummation_t<PauliString>&, const double, double);
template tuple<Array<complex_t>, Array<double>, double> KullbackLeibler::gradient_with_noise(PsiClassicalANN<2u>&, PsiCNN&, ExactSummation_t<PauliString>&, const double, double);
#endif


} // namespace ann_on_gpu

#endif // LEAN_AND_MEAN
