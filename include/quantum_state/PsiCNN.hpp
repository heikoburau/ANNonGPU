#pragma once

#include "psi_functions.hpp"

#include "bases.hpp"
#include "Array.hpp"
#include "types.h"
#include "cuda_complex.hpp"

#include <vector>
#include <list>
#include <complex>
#include <memory>


namespace ann_on_gpu {

namespace kernel {

using namespace cuda_complex;

#ifdef __CUDACC__
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif



template<typename dtype_t>
struct PsiCNN_t {
    using dtype = dtype_t;
    using real_dtype = typename cuda_complex::get_real_type<dtype>::type;

    static constexpr unsigned int max_N = MAX_SPINS;
    static constexpr unsigned int max_layers = 3u;
    static constexpr unsigned int max_channels_per_layer = 6u;
    static constexpr unsigned int max_channel_links_per_layer = 16u;

    struct Payload {
        unsigned int conf_idx;

        dtype input_activations[max_N];
        dtype output_activations[max_N];
        dtype weights[max_N];
    };


    struct Channel {
        dtype* angles;
    };

    struct ChannelLink {
        unsigned int    begin_params;
        dtype*          weights;
    };

    struct Layer {
        unsigned int  num_channels;
        unsigned int  num_channel_links;
        unsigned int  connectivity;

        Channel       channels[max_channels_per_layer];
        ChannelLink   channel_links[max_channel_links_per_layer];
    };

    unsigned int   num_sites;
    unsigned int   N;
    unsigned int   num_params;

    double         prefactor;
    dtype          log_prefactor;
    dtype          final_factor;

    Layer          layers[max_layers];
    unsigned int   num_layers;

    unsigned int   num_angles;


#ifdef __CUDACC__

    template<typename Basis_t>
    HDINLINE
    void init_payload(Payload& payload, const Basis_t& configuration, const unsigned int conf_idx) const {
        payload.conf_idx = conf_idx;
    }

    template<typename result_dtype, typename Basis_t>
    HDINLINE
    void forward_pass(
        result_dtype& result,  /* CAUTION: this parameter is assumed to be initialized */
        const Basis_t& configuration,
        Payload& payload,
        bool record_angles = false
    ) const {
        #include "cuda_kernel_defines.h"

        MULTI(j, this->N) {
            payload.input_activations[j] = configuration.network_unit_at(j);
        }

        SHARED_MEM_LOOP_BEGIN(layer_idx, this->num_layers) {
            const auto& layer = this->layers[layer_idx];

            SHARED_MEM_LOOP_BEGIN(c_j, layer.num_channels) {
                dtype REGISTER(activation, max_N);
                MULTI(i, this->N) {
                    REGISTER(activation, i) = dtype(0.0);
                }

                SHARED_MEM_LOOP_BEGIN(c_i, layer_idx > 0u ? this->layers[layer_idx - 1u].num_channels : 1u) {
                    const auto& channel_link = layer.channel_links[c_i * layer.num_channels + c_j];

                    MULTI(i, layer.connectivity) {
                        payload.weights[i] = channel_link.weights[i];
                    }
                    SYNC;

                    MULTI(j, this->N) {
                        for(auto i = 0u; i < layer.connectivity; i++) {
                            REGISTER(activation, j) += (
                                payload.weights[i] *
                                payload.input_activations[
                                    c_i * this->N + (j - layer.connectivity / 2u + i + this->N) % this->N
                                ]
                            );
                        }
                    }

                    SHARED_MEM_LOOP_END(c_i);
                }

                MULTI(j, this->N) {
                    if(record_angles) {
                        layer.channels[c_j].angles[payload.conf_idx * this->num_angles + j] = REGISTER(activation, j);
                    }

                    payload.output_activations[c_j * this->N + j] = my_logcosh(REGISTER(activation, j));
                }

                SHARED_MEM_LOOP_END(c_j);
            }

            LOOP(j, layer.num_channels * this->N) {
                if(layer_idx < this->num_layers - 1u) {
                    payload.input_activations[j] = payload.output_activations[j];
                }
                else {
                    generic_atomicAdd(&result, payload.output_activations[j] * this->final_factor);
                }
            }

            SHARED_MEM_LOOP_END(layer_idx);
        }
    }

    template<typename result_dtype, typename Basis_t>
    HDINLINE
    void log_psi_s(result_dtype& result, const Basis_t& configuration, Payload& payload) const {
        #include "cuda_kernel_defines.h"
        // CAUTION: 'result' has to be a shared variable.

        SHARED Basis_t shifted_configuration;

        SINGLE {
            result = result_dtype(this->num_sites) * result_dtype(this->log_prefactor);
            shifted_configuration = configuration;
        }
        SYNC;

        SHARED_MEM_LOOP_BEGIN(n, this->num_sites) {
            this->forward_pass(result, shifted_configuration, payload, false);

            SINGLE {
                shifted_configuration = shifted_configuration.roll(
                    1,
                    this->num_sites
                );
            }

            SHARED_MEM_LOOP_END(n);
        }
        SINGLE {
            result *= 1.0 / this->num_sites;
        }
        SYNC; // might be not neccessary
    }

    template<typename Basis_t>
    HDINLINE
    dtype psi_s(const Basis_t& configuration, Payload& payload) const {
        #include "cuda_kernel_defines.h"

        SHARED dtype log_psi;
        this->log_psi_s(log_psi, configuration, payload);

        return exp(log(this->prefactor) + log_psi);
    }

    template<typename Basis_t>
    HDINLINE void update_input_units(
        const Basis_t& old_vector, const Basis_t& new_vector, Payload& payload
    ) const {
    }

    template<typename Basis_t, typename Function>
    HDINLINE
    void foreach_O_k(const Basis_t& configuration, Payload& payload, Function function) const {
        // CAUTION: 'function' is called multiple times for each k

        #include "cuda_kernel_defines.h"

        SHARED dtype log_psi;
        this->forward_pass(log_psi, configuration, payload, true);

        LOOP(j, this->layers[this->num_layers - 1u].num_channels * this->N) {
            payload.output_activations[j] = this->final_factor;
        }

        for(auto layer_idx = int(this->num_layers) - 1; layer_idx >= 0; layer_idx--) {
            const auto& layer = this->layers[layer_idx];

            SHARED_MEM_LOOP_BEGIN(c_j, layer.num_channels) {

                MULTI(j, this->N) {
                    payload.input_activations[c_j * this->N + j] = (
                        payload.output_activations[c_j * this->N + j] *
                        my_tanh(
                            layer.channels[c_j].angles[
                                payload.conf_idx * this->num_angles + j
                            ]
                        )
                    );
                }

                SHARED_MEM_LOOP_END(c_j)
            }

            SHARED_MEM_LOOP_BEGIN(c_i, layer_idx > 0 ? this->layers[layer_idx - 1u].num_channels : 1u) {

                MULTI(i, this->N) {
                    payload.output_activations[c_i * this->N + i] = dtype(0.0);
                }

                SHARED_MEM_LOOP_BEGIN(c_j, layer.num_channels) {
                    const auto& channel_link = layer.channel_links[c_i * layer.num_channels + c_j];

                    MULTI(i, layer.connectivity) {
                        payload.weights[i] = channel_link.weights[i];
                    }
                    SYNC;

                    MULTI(j, this->N) {
                        for(auto i = 0u; i < layer.connectivity; i++) {
                            const auto lhs_unit_idx = (j - layer.connectivity / 2u + i + this->N) % this->N;

                            function(
                                channel_link.begin_params + i,
                                payload.input_activations[c_j * this->N + j] * (
                                    layer_idx == 0 ?
                                    get_real<dtype>(static_cast<real_dtype>(
                                        configuration.network_unit_at(lhs_unit_idx)
                                    )) :
                                    my_logcosh(
                                        this->layers[layer_idx - 1].channels[c_i].angles[
                                            payload.conf_idx * this->num_angles + lhs_unit_idx
                                        ]
                                    )
                                )
                            );

                            if(layer_idx > 0) {
                                generic_atomicAdd(
                                    &payload.output_activations[c_i * this->N + lhs_unit_idx],
                                    payload.weights[i] *
                                    payload.input_activations[c_j * this->N + j]
                                );
                            }
                        }
                    }

                    SHARED_MEM_LOOP_END(c_j);
                }

                SHARED_MEM_LOOP_END(c_i);
            }
        }
    }

#endif // __CUDACC__

    const PsiCNN_t& kernel() const {
        return *this;
    }

    PsiCNN_t& kernel() {
        return *this;
    }

    HDINLINE
    unsigned int get_width() const {
        return this->N;
    }

    HDINLINE
    unsigned int get_num_angles() const {
        return 0u;
    }

    HDINLINE unsigned int get_num_input_units() const {
        return this->N;
    }

    HDINLINE
    double probability_s(const double log_psi_s_real) const {
        return exp(2.0 * (log(this->prefactor) + log_psi_s_real));
    }

};

} // namespace kernel


template<typename dtype>
struct PsiCNN_t : public kernel::PsiCNN_t<dtype> {

    using real_dtype = typename cuda_complex::get_real_type<dtype>::type;
    using Kernel = kernel::PsiCNN_t<dtype>;

    bool gpu;
    Array<unsigned int> num_channels_list;
    Array<unsigned int> connectivity_list;
    Array<dtype> params;
    Array<dtype> angles;

#ifdef __PYTHONCC__

    inline PsiCNN_t(
        const unsigned int num_sites,
        const unsigned int N,
        const xt::pytensor<unsigned int, 1u>& num_channels_list,
        const xt::pytensor<unsigned int, 1u>& connectivity_list,
        const xt::pytensor<dtype, 1u>& params,
        const dtype& final_factor,
        const double prefactor,
        const bool gpu
    )
    :
    gpu(gpu),
    num_channels_list(num_channels_list, false),
    connectivity_list(connectivity_list, false),
    params(params, gpu),
    angles(1, gpu)
    {
        this->num_sites = num_sites;
        this->N = N;
        this->final_factor = final_factor;
        this->prefactor = prefactor;
        this->log_prefactor = dtype(0.0);

        this->init_kernel();
    }

#endif // __PYTHONCC__

    inline PsiCNN_t(const PsiCNN_t& other)
    :
    gpu(other.gpu),
    num_channels_list(other.num_channels_list),
    connectivity_list(other.connectivity_list),
    params(other.params),
    angles(other.angles)
    {
        this->num_sites = other.num_sites;
        this->N = other.N;
        this->final_factor = other.final_factor;
        this->prefactor = other.prefactor;
        this->log_prefactor = other.log_prefactor;

        this->init_kernel();
    }

    PsiCNN_t copy() const {
        return *this;
    }

    template<typename Ensemble>
    void init(const Ensemble& ensemble);

    void init_kernel();
};


using PsiCNN = PsiCNN_t<complex_t>;

} // namespace ann_on_gpu
