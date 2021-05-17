#ifdef ENABLE_PSI_DEEP

#include "quantum_state/PsiDeep.hpp"
#include "ensembles.hpp"

#include <complex>
#include <vector>
#include <random>
#include <cstring>
#include <algorithm>
#include <iterator>


namespace ann_on_gpu {

using namespace cuda_complex;

template<typename dtype, bool symmetric>
PsiDeepT<dtype, symmetric>::PsiDeepT(const unsigned int N, const unsigned int M, const bool gpu)
    :
    input_weights(N, gpu),
    final_weights(M, gpu),
    gpu(gpu)
{
    this->num_sites = N;
    this->N = N;
    this->log_prefactor = dtype(0.0);
    this->num_layers = 2u;
    this->width = this->N;
    this->num_units = 0u;

    for(auto layer_idx = int(this->num_layers) - 1; layer_idx > 0; layer_idx--) {
        const unsigned int size = M;
        // const unsigned int connectivity = N;

        if(size > this->width) {
            this->width = size;
        }

        this->num_units += size;

        Array<unsigned int> connections_array(N * M, gpu);
        Array<dtype> weights_array(N * M, gpu);
        Array<dtype> biases_array(M, gpu);


        for(auto j = 0u; j < M; j++) {
            for(auto i = 0u; i < N; i++) {
                connections_array[i * M + j] = i;
                weights_array[i * M + j] = dtype(0.0);
                if(i == (j % N)) {
                    weights_array[i * M + j] = dtype(0.01);
                }
            }
            biases_array[j] = dtype(0.0);
        }
        connections_array.update_device();
        weights_array.update_device();
        biases_array.update_device();

        this->layers.push_front({
            size,
            N,
            move(connections_array),
            move(weights_array),
            move(biases_array)
        });
    }
    // input layer (spins)
    this->layers.push_front({
        this->N,
        0u,
        move(Array<unsigned int>(1, gpu)),
        move(Array<dtype>(1, gpu)),
        move(Array<dtype>(1, gpu))
    });

    this->input_weights.clear();

    this->init_kernel();
}

template<typename dtype, bool symmetric>
PsiDeepT<dtype, symmetric>::PsiDeepT(const PsiDeepT<dtype, symmetric>& other)
    :
    layers(other.layers),
    input_weights(other.input_weights),
    final_weights(other.final_weights),
    gpu(other.gpu)
{
    this->num_sites = other.num_sites;
    this->N = other.N;
    this->log_prefactor = other.log_prefactor;
    this->num_layers = other.num_layers;
    this->width = other.width;
    this->num_units = other.num_units;
    this->num_final_weights = other.num_final_weights;

    this->init_kernel();
}

template<typename dtype, bool symmetric>
PsiDeepT<dtype, symmetric>& PsiDeepT<dtype, symmetric>::operator=(const PsiDeepT<dtype, symmetric>& other) {
    this->layers = other.layers;
    this->input_weights = other.input_weights;
    this->final_weights = other.final_weights;
    this->gpu = other.gpu;

    this->num_sites = other.num_sites;
    this->N = other.N;
    this->log_prefactor = other.log_prefactor;
    this->num_layers = other.num_layers;
    this->width = other.width;
    this->num_units = other.num_units;
    this->num_final_weights = other.num_final_weights;

    this->init_kernel();

    return *this;
}


template<typename dtype, bool symmetric>
void PsiDeepT<dtype, symmetric>::init_kernel() {
    #ifdef ENABLE_NETWORK_BASES
    this->num_params = this->N; // initial biases
    #else
    this->num_params = 0u;
    #endif //ENABLE_NETWORK_BASES

    auto angle_idx = 0u;
    for(auto layer_idx = 0u; layer_idx < this->num_layers; layer_idx++) {
        const auto& layer = *next(this->layers.begin(), layer_idx);
        auto& kernel_layer = kernel::PsiDeepT<dtype, symmetric>::layers[layer_idx];
        kernel_layer.size = layer.size;
        kernel_layer.connectivity = layer.connectivity;

        if(layer_idx > 0u) { // input layer has no parameters (weight matrix)
            kernel_layer.begin_params = this->num_params;
            #ifdef ENABLE_NETWORK_BASES
            this->num_params += layer.size;
            #endif //ENABLE_NETWORK_BASES
            this->num_params += layer.weights.size();

            if(layer_idx > 1u) {
                kernel_layer.begin_deep_angles = angle_idx;
                angle_idx += layer.size;
            }
        }
    }
    this->num_final_weights = this->layers.back().size;

    this->update_kernel();
}


template<typename dtype, bool symmetric>
void PsiDeepT<dtype, symmetric>::update_kernel() {
    for(auto layer_idx = 0u; layer_idx < this->num_layers; layer_idx++) {
        Layer& layer = *next(this->layers.begin(), layer_idx);
        auto& kernel_layer = this->kernel().layers[layer_idx];

        layer.connections.update_device();
        layer.weights.update_device();
        layer.biases.update_device();

        kernel_layer.connections = layer.connections.data();
        kernel_layer.weights = layer.weights.data();
        kernel_layer.biases = layer.biases.data();
    }

    this->input_weights.update_device();
    this->final_weights.update_device();

    this->kernel().input_weights = this->input_weights.data();
    this->kernel().final_weights = this->final_weights.data();
}


template<typename dtype, bool symmetric>
Array<dtype> PsiDeepT<dtype, symmetric>::get_params() const {
    Array<dtype> result(this->num_params, false);
    auto it = result.begin();

    #ifdef ENABLE_NETWORK_BASES
    for(auto i = 0u; i < this->N; i++) {
        result[i] = this->input_weights[i];
    }
    it += this->N;
    #endif // ENABLE_NETWORK_BASES

    for(auto layer_it = next(this->layers.begin()); layer_it != this->layers.end(); layer_it++) {
        auto& layer = *layer_it;

        #ifdef ENABLE_NETWORK_BASES
        copy(layer.biases.begin(), layer.biases.end(), it);
        it += layer.biases.size();
        #endif // ENABLE_NETWORK_BASES

        copy(layer.weights.begin(), layer.weights.end(), it);
        it += layer.weights.size();
    }

    return result;
}


template<typename dtype, bool symmetric>
void PsiDeepT<dtype, symmetric>::set_params(const Array<dtype>& new_params) {
    auto it = new_params.begin();

    #ifdef ENABLE_NETWORK_BASES
    for(auto i = 0u; i < this->N; i++) {
        this->input_weights[i] = new_params[i];
    }
    it += this->N;
    #endif // ENABLE_NETWORK_BASES

    for(auto layer_it = next(this->layers.begin()); layer_it != this->layers.end(); layer_it++) {
        auto& layer = *layer_it;

        #ifdef ENABLE_NETWORK_BASES
        copy(it, it + layer.biases.size(), layer.biases.begin());
        it += layer.size;
        #endif // ENABLE_NETWORK_BASES

        copy(it, it + layer.weights.size(), layer.weights.begin());
        it += layer.weights.size();
    }

    this->update_kernel();
}


// template struct PsiDeepT<cuda_complex::complex<float>>;

#ifdef PSI_DEEP_SYMMETRIC
// template struct PsiDeepT<cuda_complex::complex<double>, true>;
template struct PsiDeepT<double, true>;
template struct PsiDeepT<complex_t, true>;
#else
template struct PsiDeepT<double, false>;
template struct PsiDeepT<complex_t, false>;
#endif // PSI_DEEP_SYMMETRIC

} // namespace ann_on_gpu

#endif // ENABLE_PSI_DEEP
