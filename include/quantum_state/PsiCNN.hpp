#pragma once

#include "psi_functions.hpp"
#include "detail/Convolve.hpp"

#include "bases.hpp"
#include "Array.hpp"
#include "types.h"
#include "cuda_complex.hpp"

#include <vector>
#include <array>
#include <list>
#include <set>
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


template<unsigned int dim_t, typename dtype_t>
struct PsiCNN_t {
    static constexpr auto dim = dim_t;
    using dtype = dtype_t;
    using real_dtype = typename cuda_complex::get_real_type<dtype>::type;

    static constexpr auto max_N = MAX_SPINS;
    static constexpr auto max_layers = 3u;
    static constexpr auto max_channels_per_layer = 6u;
    static constexpr auto max_channel_links_per_layer = 9u;
    static constexpr auto max_symmetry_classes = 4u;

    struct Payload {
        unsigned int conf_idx;

        dtype input_activations[max_N];
        dtype output_activations[max_N];
        dtype weights[max_symmetry_classes * max_N];
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
        unsigned int  connectivity[dim];
        unsigned int  connectivity_vol;

        Channel       channels[max_channels_per_layer];
        ChannelLink   channel_links[max_channel_links_per_layer];
    };

    unsigned int   N;
    unsigned int   num_sites;
    unsigned int   num_params;
    unsigned int   num_symmetry_classes;

    detail::Convolve<dim> convolve;

    dtype          log_prefactor;
    double         final_factor;

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

                    LOOP(i, this->num_symmetry_classes * layer.connectivity_vol) {
                        payload.weights[i] = channel_link.weights[i];
                    }
                    SYNC;

                    MULTI(j, this->N) {
                        REGISTER(activation, j) += this->convolve(
                            j, layer.connectivity, layer.connectivity_vol,
                            payload.weights, payload.input_activations + c_i * this->N
                        );
                    }

                    SHARED_MEM_LOOP_END(c_i);
                }

                MULTI(j, this->N) {
                    if(record_angles) {
                        layer.channels[c_j].angles[payload.conf_idx * this->num_angles + j] = REGISTER(activation, j);
                    }

                    payload.output_activations[c_j * this->N + j] = my_logcosh(REGISTER(activation, j), layer_idx);
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

        SINGLE {
            result = result_dtype(this->log_prefactor);
        }
        SYNC;

        this->forward_pass(result, configuration, payload, false);
        SYNC;
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
            payload.output_activations[j] = dtype(this->final_factor);
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
                            ],
                            layer_idx
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

                    LOOP(i, this->num_symmetry_classes * layer.connectivity_vol) {
                        payload.weights[i] = channel_link.weights[i];
                    }
                    SYNC;

                    MULTI(j, this->N) {
                        this->convolve.foreach_connection(
                            j, layer.connectivity,
                            [&](const unsigned int conn_idx, const unsigned int input_idx) {
                                function(
                                    channel_link.begin_params + this->convolve.symmetry_classes[j] * layer.connectivity_vol + conn_idx,
                                    payload.input_activations[c_j * this->N + j] * (
                                        layer_idx == 0 ?
                                        dtype(configuration.network_unit_at(input_idx)) :
                                        my_logcosh(
                                            this->layers[layer_idx - 1].channels[c_i].angles[
                                                payload.conf_idx * this->num_angles + input_idx
                                            ],
                                            layer_idx - 1
                                        )
                                    )
                                );

                                if(layer_idx > 0) {
                                    generic_atomicAdd(
                                        &payload.output_activations[c_i * this->N + input_idx],
                                        payload.weights[this->convolve.symmetry_classes[j] * layer.connectivity_vol + conn_idx] *
                                        payload.input_activations[c_j * this->N + j]
                                    );
                                }
                            }
                        );
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

    HDINLINE unsigned int get_num_input_units() const {
        return this->N;
    }
};

} // namespace kernel


template<unsigned int dim_t, typename dtype>
struct PsiCNN_t : public kernel::PsiCNN_t<dim_t, dtype> {
    static constexpr auto dim = dim_t;
    using std_dtype = typename std_dtype<dtype>::type;
    using real_dtype = typename cuda_complex::get_real_type<dtype>::type;
    using Kernel = kernel::PsiCNN_t<dim, dtype>;

    bool gpu;
    std::array<unsigned int, dim> extent;
    Array<unsigned int> num_channels_list;
    Array<unsigned int> connectivity_list;
    Array<unsigned int> symmetry_classes;
    Array<dtype> params;
    Array<dtype> angles;

#ifdef __PYTHONCC__

    inline PsiCNN_t(
        const array<unsigned int, dim>& extent,
        const xt::pytensor<unsigned int, 1u>& num_channels_list,
        const xt::pytensor<unsigned int, 2u>& connectivity_list,
        const xt::pytensor<unsigned int, 1u>& symmetry_classes,
        const xt::pytensor<std_dtype, 1u>& params,
        const double final_factor,
        const std::complex<double> log_prefactor,
        const bool gpu
    ):
        gpu(gpu),
        extent(extent),
        num_channels_list(num_channels_list, false),
        connectivity_list(connectivity_list, false),
        symmetry_classes(symmetry_classes, false),
        params(params, gpu),
        angles(gpu)
    {
        auto N = 1u;
        for(auto d = 0u; d < dim; d++) {
            N *= extent[d];
        }
        this->N = N;
        this->num_sites = N;
        this->final_factor = final_factor;
        this->log_prefactor = log_prefactor;

        this->num_symmetry_classes = std::set<unsigned int>(
            symmetry_classes.begin(), symmetry_classes.end()
        ).size();

        this->init_kernel();
    }

#endif // __PYTHONCC__

    inline PsiCNN_t(const PsiCNN_t& other):
        gpu(other.gpu),
        extent(other.extent),
        num_channels_list(other.num_channels_list),
        connectivity_list(other.connectivity_list),
        symmetry_classes(other.symmetry_classes),
        params(other.params),
        angles(other.angles)
    {
        this->N = other.N;
        this->num_sites = other.num_sites;
        this->final_factor = other.final_factor;
        this->log_prefactor = other.log_prefactor;

        this->num_symmetry_classes = other.num_symmetry_classes;

        this->init_kernel();
    }

    PsiCNN_t copy() const {
        return *this;
    }

    void init_gradient(const unsigned int num_steps);
    void init_kernel_angles();

    void init_kernel();
};


using PsiCNN = PsiCNN_t<3u, complex_t>;

} // namespace ann_on_gpu
