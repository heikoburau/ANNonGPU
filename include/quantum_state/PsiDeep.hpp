#pragma once

#include "psi_functions.hpp"

#include "RNGStates.hpp"
#include "bases.hpp"
#include "Array.hpp"
#include "types.h"
#ifdef __CUDACC__
    #include "utils.kernel"
#endif
#include "cuda_complex.hpp"

#include <vector>
#include <list>
#include <complex>
#include <memory>

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"

    using namespace std::complex_literals;
#endif // __PYTHONCC__


// #define DIM 1


namespace ann_on_gpu {

namespace kernel {

using namespace cuda_complex;

#ifdef __CUDACC__
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

namespace PsiDeep {

    template<typename dtype, unsigned int, bool>
    struct Payload_t {};

    template<typename dtype, unsigned int max_width>
    struct Payload_t<dtype, max_width, false> {
        dtype angles[max_width];
        dtype activations[max_width];

        // curandState_t rng_state;
    };

    template<typename dtype, unsigned int max_width>
    struct Payload_t<dtype, max_width, true> {
        dtype angles[max_width];
        dtype activations[max_width];



        // struct {
        //     char data[sizeof(curandState_t) + sizeof(mt19937)];
        // curandState_t rng_state;
    };

} // namespace PsiDeep


template<typename dtype_t, bool symmetric>
struct PsiDeepT {
    // network structure:
    //
    //  layers:
    //      - 1st layer: input (spins)
    //      - ... hidden layers
    //  sum up units to a single unit (log psi) with final_weights (this does not count as a layer here)


    using dtype = dtype_t;
    using real_dtype = typename cuda_complex::get_real_type<dtype>::type;

    static constexpr unsigned int max_N = 50u;
    static constexpr unsigned int max_layers = 4u;
    static constexpr unsigned int max_width = max_N;
    static constexpr unsigned int max_deep_angles = max_N;


    using Payload = PsiDeep::Payload_t<dtype, max_width, symmetric>;

    // TODO: Try to use stack-allocated arrays
    struct Layer {
        unsigned int  size;                 // number of units
        unsigned int  begin_deep_angles;         // index of the first unit of this layer in a global list of angles
        unsigned int  begin_params;         // index of the first unit of this layer in a global list of parameters
        unsigned int  lhs_connectivity;     // number of connections to the lhs per unit
        unsigned int  rhs_connectivity;     // number of connections to the rhs per unit
        unsigned int* RESTRICT lhs_connections;      // connectivity matrix to the lhs: lhs-connectivity x size
        unsigned int* RESTRICT rhs_connections;      // connectivity matrix to the rhs: size x rhs-connectivity
        dtype*    RESTRICT lhs_weights;          // weight matrix to the lhs: lhs-connectivity x size, var.parameters
        dtype*    RESTRICT rhs_weights;          // weight matrix to the rhs: size x rhs-connectivity, var.parameters
        dtype*    RESTRICT biases;               // bias factors, var.parameters

        HDINLINE unsigned int lhs_connection(const unsigned int i, const unsigned int j) const {
            return this->lhs_connections[i * this->size + j];
        }
        HDINLINE unsigned int rhs_connection(const unsigned int i, const unsigned int j) const {
            return this->rhs_connections[i * this->rhs_connectivity + j];
        }
        HDINLINE dtype lhs_weight(const unsigned int i, const unsigned int j) const {
            return this->lhs_weights[i * this->size + j];
        }
        HDINLINE dtype rhs_weight(const unsigned int i, const unsigned int j) const {
            return this->rhs_weights[i * this->rhs_connectivity + j];
        }
    };

    unsigned int   N;
    unsigned int   num_sites;

    unsigned int   num_params;

    dtype          log_prefactor;

    dtype*         RESTRICT input_weights;
    Layer          layers[max_layers];
    dtype*         RESTRICT final_weights;
    unsigned int   num_final_weights;
    unsigned int   num_layers;
    unsigned int   width;                   // size of largest layer
    unsigned int   num_units;

    unsigned int   N_i;
    unsigned int   N_j;

#ifdef __CUDACC__

#ifdef ENABLE_SPINS
    HDINLINE
    void compute_angles(dtype* angles, const Spins& configuration) const {
    }
#endif // ENABLE_SPINS

    HDINLINE
    void compute_angles(dtype* angles, const complex_t* configuration) const {
        const Layer& layer = this->layers[1];

        MULTI(j, layer.size) {
            angles[j] = dtype(0.0);

            for(auto i = 0u; i < layer.lhs_connectivity; i++) {
                angles[j] += layer.lhs_weight(i, j) * configuration[layer.lhs_connection(i, j)];
            }

            #ifdef ENABLE_NETWORK_BASES
            angles[j] += layer.biases[j];
            #endif // ENABLE_NETWORK_BASES
        }
        SYNC;
    }

    template<typename Basis_t>
    HDINLINE
    void init_payload(Payload& payload, const Basis_t& configuration, const unsigned int conf_idx) const {
        // if(!symmetric) {
        //     this->compute_angles(payload.angles, configuration);
        // }
    }

    HDINLINE
    void save_payload(Payload& payload) const {
    }

    template<typename result_dtype>
    HDINLINE
    void forward_pass(
        result_dtype& result,  /* CAUTION: this parameter is assumed to be initialized */
        dtype* RESTRICT angles,
        dtype* RESTRICT activations,
        dtype* RESTRICT deep_angles
    ) const {
        #include "cuda_kernel_defines.h"

        MULTI(i, this->layers[1].size) {
            activations[i] = my_logcosh(angles[i], 0u);
        }

        SHARED_MEM_LOOP_BEGIN_X0(layer_idx, 2u, this->num_layers) {
            const Layer& layer = this->layers[layer_idx];
            dtype REGISTER(activation, max_width);
            MULTI(j, layer.size) {
                REGISTER(activation, j) = dtype(0.0);

                for(auto i = 0u; i < layer.lhs_connectivity; i++) {
                    REGISTER(activation, j) += (
                        layer.lhs_weight(i, j) *
                        activations[layer.lhs_connection(i, j)]
                    );
                }
                #ifdef ENABLE_NETWORK_BASES
                REGISTER(activation, j) += layer.biases[j];
                #endif // ENABLE_NETWORK_BASES

                if(deep_angles != nullptr) {
                    deep_angles[layer.begin_deep_angles + j] = REGISTER(activation, j);
                }
            }
            SYNC;
            MULTI(k, layer.size) {
                activations[k] = my_logcosh(REGISTER(activation, k), layer_idx - 1u);
            }
            SHARED_MEM_LOOP_END(layer_idx);
        }
        MULTI(j, this->num_final_weights) {
            generic_atomicAdd(&result, activations[j] * this->final_weights[j]);
        }
        SYNC;
    }

    template<typename result_dtype, typename Basis_t>
    HDINLINE
    void log_psi_s(result_dtype& result, const Basis_t& configuration, Payload& payload) const {
        #include "cuda_kernel_defines.h"
        // CAUTION: 'result' has to be a shared variable.

        // SHARED Basis_t shifted_configuration;

        SINGLE {
            result = result_dtype(this->log_prefactor);

            // if(symmetric) {
            //     shifted_configuration = configuration;
            //     result *= this->num_sites;
            // }
        }
        this->compute_angles(payload.angles, configuration);

        // MULTI(i, this->N) {
        //     generic_atomicAdd(&result, result_dtype(configuration.network_unit_at(i)) * this->input_weights[i]);
        // }

        this->forward_pass(result, payload.angles, payload.activations, nullptr);

        // printf("%f, %f\n", result.real(), result.imag());
    }


    template<typename Basis_t>
    HDINLINE void update_input_units(
        const Basis_t& old_vector, const Basis_t& new_vector, Payload& payload
    ) const {
        #include "cuda_kernel_defines.h"

    }

#ifdef ENABLE_SPINS
    template<typename Function>
    HDINLINE
    void foreach_O_k(const Spins& configuration, Payload& payload, Function function) const {
    }
#endif // ENABLE_SPINS

    template<typename Function>
    HDINLINE
    void foreach_O_k(const complex_t* configuration, Payload& payload, Function function) const {
        #include "cuda_kernel_defines.h"

        SHARED dtype deep_angles[max_deep_angles];
        SHARED dtype log_psi;

        #ifdef ENABLE_NETWORK_BASES
        MULTI(i, this->N) {
            function(
                i,
                configuration[i]
            );
        }
        #endif // ENABLE_NETWORK_BASES

        this->compute_angles(payload.angles, configuration);
        // note: since log_psi isn't needed here, it doesn't need to be initialized too
        this->forward_pass(log_psi, payload.angles, payload.activations, deep_angles);

        for(int layer_idx = int(this->num_layers) - 1; layer_idx > 0; layer_idx--) {
            const Layer& layer = this->layers[layer_idx];

            // calculate the activations of the layer.
            // here, these are the back-propagated derivatives.
            if(layer_idx == this->num_layers - 1) {
                MULTI(j, layer.size) {
                    payload.activations[j] = this->final_weights[j] * (
                        layer_idx == 1 ?
                        my_tanh(payload.angles[j], 0u) :
                        my_tanh(deep_angles[
                            layer.begin_deep_angles + j
                        ], layer_idx - 1)
                    );
                }
            } else {
                // TODO: check if shared memory solution is faster (most likely not)
                dtype REGISTER(unit_activation, max_width);

                SYNC;
                MULTI(i, layer.size) {
                    REGISTER(unit_activation, i) = dtype(0.0);

                    for(auto j = 0u; j < layer.rhs_connectivity; j++) {
                        REGISTER(unit_activation, i) += (
                            layer.rhs_weight(i, j) * payload.activations[
                                layer.rhs_connection(i, j)
                            ]
                        );
                    }
                    REGISTER(unit_activation, i) *= (
                        layer_idx == 1 ?
                        my_tanh(payload.angles[i], 0u) :
                        my_tanh(deep_angles[layer.begin_deep_angles + i], layer_idx - 1)
                    );
                }
                SYNC;
                MULTI(j, layer.size) {
                    payload.activations[j] = REGISTER(unit_activation, j);
                }
            }
            MULTI(j, layer.size) {
                #ifdef ENABLE_NETWORK_BASES
                function(layer.begin_params + j, payload.activations[j]);
                #endif // ENABLE_NETWORK_BASES

                for(auto i = 0u; i < layer.lhs_connectivity; i++) {
                    const auto lhs_unit_idx = layer.lhs_connection(i, j);
                    // TODO: check if shared memory solution is faster

                    function(
                        #ifdef ENABLE_NETWORK_BASES
                        layer.begin_params + layer.size + i * layer.size + j,
                        #else
                        layer.begin_params + i * layer.size + j,
                        #endif // ENABLE_NETWORK_BASES
                        payload.activations[j] * (
                            layer_idx == 1 ?
                            configuration[lhs_unit_idx] :
                            (
                                layer_idx == 2 ?
                                my_logcosh(payload.angles[lhs_unit_idx], 0u) :
                                my_logcosh(deep_angles[
                                    this->layers[layer_idx - 1].begin_deep_angles + lhs_unit_idx
                                ], layer_idx - 2)
                            )
                        )
                    );
                }
            }
        }
        MULTI(j, this->num_final_weights) {
            function(
                this->num_params - this->num_final_weights + j,
                this->num_layers == 2u ?
                my_logcosh(payload.angles[j], 0u) :
                my_logcosh(deep_angles[this->layers[this->num_layers - 1].begin_deep_angles + j], this->num_layers - 2u)
            );
        }
    }

#endif // __CUDACC__

    const PsiDeepT& kernel() const {
        return *this;
    }

    PsiDeepT& kernel() {
        return *this;
    }

    HDINLINE
    unsigned int get_width() const {
        return this->width;
    }

    HDINLINE unsigned int get_num_input_units() const {
        return this->N;
    }
};

} // namespace kernel


template<typename dtype, bool symmetric>
struct PsiDeepT : public kernel::PsiDeepT<dtype, symmetric> {

    using real_dtype = typename cuda_complex::get_real_type<dtype>::type;
    using Kernel = kernel::PsiDeepT<dtype, symmetric>;

    struct Layer {
        unsigned int        size;
        unsigned int        lhs_connectivity;
        Array<unsigned int> lhs_connections;
        Array<unsigned int> rhs_connections;
        Array<dtype>        lhs_weights;
        Array<dtype>        rhs_weights;
        Array<dtype>        biases;
    };
    list<Layer>  layers;
    Array<dtype> input_weights;
    Array<dtype> final_weights;
    bool         gpu;

    PsiDeepT(const unsigned int N, const unsigned int M, const bool gpu);
    PsiDeepT(const PsiDeepT& other);
    PsiDeepT& operator=(const PsiDeepT& other);

#ifdef __PYTHONCC__

    inline PsiDeepT(
        const unsigned int num_sites,
        const xt::pytensor<typename std_dtype<dtype>::type, 1u>& input_weights,
        const vector<xt::pytensor<typename std_dtype<dtype>::type, 1u>> biases_list,
        const vector<xt::pytensor<unsigned int, 2u>>& lhs_connections_list,
        const vector<xt::pytensor<typename std_dtype<dtype>::type, 2u>>& lhs_weights_list,
        const xt::pytensor<typename std_dtype<dtype>::type, 1u>& final_weights,
        const double log_prefactor,
        const bool gpu
    ) : input_weights(input_weights, gpu), final_weights(final_weights, gpu) {
        this->num_sites = num_sites;
        this->N = input_weights.shape()[0];
        this->log_prefactor = log_prefactor;
        this->num_layers = lhs_weights_list.size() + 1u; // num hidden layers + input layer
        this->width = this->N;
        this->num_units = 0u;
        this->gpu = gpu;

        Array<unsigned int> rhs_connections_array(0, false);
        Array<dtype> rhs_weights_array(0, false);

        for(auto layer_idx = int(this->num_layers) - 1; layer_idx > 0; layer_idx--) {
            const auto& lhs_connections = lhs_connections_list[layer_idx - 1];
            const auto& lhs_weights = lhs_weights_list[layer_idx - 1];
            const auto& biases = biases_list[layer_idx - 1];

            const unsigned int size = biases.size();
            const unsigned int lhs_connectivity = lhs_connections.shape()[0];

            if(size > this->width) {
                this->width = size;
            }

            this->num_units += size;

            Array<unsigned int> lhs_connections_array(lhs_connections, gpu);
            Array<dtype> lhs_weights_array(lhs_weights, gpu);
            Array<dtype> biases_array(biases, gpu);

            // WARNING: do not make this const! Otherwise its content will be copied on assignment, not moved.
            auto rhs_connections_and_weights = this->compile_rhs_connections_and_weights(
                layer_idx > 1 ? biases_list[layer_idx - 2].size() : this->N,
                size,
                lhs_connectivity,
                lhs_connections_array,
                lhs_weights_array
            );


            this->layers.push_front({
                size,
                lhs_connectivity,
                move(lhs_connections_array),
                move(rhs_connections_array),
                move(lhs_weights_array),
                move(rhs_weights_array),
                move(biases_array)
            });

            rhs_connections_array = move(rhs_connections_and_weights.first);
            rhs_weights_array = move(rhs_connections_and_weights.second);
        }

        // input layer (spins)
        this->layers.push_front({
            this->N,
            0u,
            move(Array<unsigned int>(1, gpu)),
            move(rhs_connections_array),
            move(Array<dtype>(1, gpu)),
            move(rhs_weights_array),
            move(Array<dtype>(1, gpu))
        });

        this->init_kernel();

        // cout << "N: " << this->N << endl;
        // cout << "num_layers: " << this->num_layers << endl;
        // cout << "width: " << this->width << endl;
        // cout << "num_params: " << this->num_params << endl;
        // cout << "log_prefactor: " << this->log_prefactor << endl;
        // cout << "num_final_weights: " << this->num_final_weights << endl;
        // cout << endl;

        // for(auto layer_idx = int(this->num_layers) - 1; layer_idx >= 0; layer_idx--) {
        //     const auto& kernel_layer = kernel::PsiDeepT<dtype, symmetric>::layers[layer_idx];
        //     const auto& layer = *next(this->layers.begin(), layer_idx);

        //     cout << "Layer: " << layer_idx << endl;
        //     cout << "size: " << kernel_layer.size << endl;
        //     cout << "lhs_connectivity: " << kernel_layer.lhs_connectivity << endl;
        //     cout << "rhs_connectivity: " << kernel_layer.rhs_connectivity << endl;
        //     cout << "begin_params: " << kernel_layer.begin_params << endl;
        //     cout << "begin_deep_angles: " << kernel_layer.begin_deep_angles << endl;
        //     cout << "lhs_weights.size: " << layer.lhs_weights.size() << endl;
        //     cout << "rhs_weights.size: " << layer.rhs_weights.size() << endl;
        //     cout << "biases.size: " << layer.biases.size() << endl;
        //     cout << "rhs_connections.size: " << layer.rhs_connections.size() << endl;
        //     cout << "lhs_connections: " << endl;
        //     for(auto i = 0u; i < layer.lhs_connectivity; i++) {
        //         for(auto j = 0u; j < layer.size; j++) {
        //             cout << layer.lhs_connections[i * layer.size + j] << ", ";
        //         }
        //         cout << endl;
        //     }
        //     cout << "rhs_connections: " << endl;
        //     for(auto i = 0u; i < layer.size; i++) {
        //         for(auto j = 0u; j < kernel_layer.rhs_connectivity; j++) {
        //             cout << layer.rhs_connections[i * kernel_layer.rhs_connectivity + j] << ", ";
        //         }
        //         cout << endl;
        //     }
        //     cout << endl;
        // }
    }

    PsiDeepT copy() const {
        return *this;
    }

#ifdef ENABLE_SPINS
    xt::pytensor<complex<double>, 1> O_k_vector_py(const Spins& spins) {
        return psi_O_k_vector_py(*this, spins);
    }
#endif

    inline vector<xt::pytensor<typename std_dtype<dtype>::type, 1u>> get_b() const {
        vector<xt::pytensor<typename std_dtype<dtype>::type, 1u>> result;

        for(auto layer_it = next(this->layers.begin()); layer_it != this->layers.end(); layer_it++) {
            auto& layer = *layer_it;
        // for(const auto& layer : this->layers) {
            result.push_back(layer.biases.to_pytensor_1d());
        }

        return result;
    }

    inline vector<xt::pytensor<typename std_dtype<dtype>::type, 2u>> get_W() const {
        vector<xt::pytensor<typename std_dtype<dtype>::type, 2u>> result;

        for(auto layer_it = next(this->layers.begin()); layer_it != this->layers.end(); layer_it++) {
            auto& layer = *layer_it;
        // for(const auto& layer : this->layers) {
            result.push_back(layer.lhs_weights.to_pytensor_2d(shape_t<2u>{
                (long int)layer.lhs_connectivity, (long int)layer.size
            }));
        }

        return result;
    }

    inline vector<xt::pytensor<unsigned int, 2>> get_connections() const {
        vector<xt::pytensor<unsigned int, 2>> result;

        for(auto layer_it = next(this->layers.begin()); layer_it != this->layers.end(); layer_it++) {
            auto& layer = *layer_it;
        // for(const auto& layer : this->layers) {
            result.push_back(layer.lhs_connections.to_pytensor_2d(shape_t<2u>{
                (long int)layer.lhs_connectivity, (long int)layer.size
            }));
        }

        return result;
    }

#endif // __PYTHONCC__

    Array<dtype> get_params() const;
    void set_params(const Array<dtype>& new_params);

    inline bool is_symmetric() const {
        return symmetric;
    }

    void init_kernel();
    void update_kernel();

    pair<Array<unsigned int>, Array<dtype>> compile_rhs_connections_and_weights(
        const unsigned int prev_size,
        const unsigned int size,
        const unsigned int lhs_connectivity,
        const Array<unsigned int>& lhs_connections,
        const Array<dtype>& lhs_weights
    );
};


#ifdef PSI_DEEP_SYMMETRIC
using PsiDeep = PsiDeepT<complex_t, true>;
#else
using PsiDeep = PsiDeepT<complex_t, false>;
#endif // PSI_DEEP_SYMMETRIC

// using PsiDeep = PsiDeepT<cuda_complex::complex<double>>;

} // namespace ann_on_gpu
