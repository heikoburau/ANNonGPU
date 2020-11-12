#include "quantum_state/PsiDeep.hpp"

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
    this->prefactor = 1.0;
    this->log_prefactor = dtype(0.0);
    this->num_layers = 2u;
    this->width = this->N;
    this->num_units = 0u;

    Array<unsigned int> rhs_connections_array(0, false);
    Array<dtype> rhs_weights_array(0, false);

    for(auto layer_idx = int(this->num_layers) - 1; layer_idx > 0; layer_idx--) {
        const unsigned int size = M;
        const unsigned int lhs_connectivity = N;

        if(size > this->width) {
            this->width = size;
        }

        this->num_units += size;

        Array<unsigned int> lhs_connections_array(N * M, gpu);
        Array<dtype> lhs_weights_array(N * M, gpu);
        Array<dtype> biases_array(M, gpu);


        for(auto j = 0u; j < M; j++) {
            for(auto i = 0u; i < N; i++) {
                lhs_connections_array[i * M + j] = i;
                lhs_weights_array[i * M + j] = dtype(0.0);
                if(i == (j % N)) {
                    lhs_weights_array[i * M + j] = dtype(0.01);
                }
            }
            biases_array[j] = dtype(0.0);
        }
        lhs_connections_array.update_device();
        lhs_weights_array.update_device();
        biases_array.update_device();


        const auto rhs_connections_and_weights = this->compile_rhs_connections_and_weights(
            this->N,
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
    this->prefactor = other.prefactor;
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
    this->prefactor = other.prefactor;
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
    this->num_params = this->N; // initial biases
    auto angle_idx = 0u;
    for(auto layer_idx = 0u; layer_idx < this->num_layers; layer_idx++) {
        const auto& layer = *next(this->layers.begin(), layer_idx);
        auto& kernel_layer = kernel::PsiDeepT<dtype, symmetric>::layers[layer_idx];
        kernel_layer.size = layer.size;
        kernel_layer.lhs_connectivity = layer.lhs_connectivity;

        if(layer_idx > 0u) { // input layer has no parameters (weight matrix)
            kernel_layer.begin_params = this->num_params;
            this->num_params += layer.size + layer.lhs_weights.size();

            if(layer_idx > 1u) {
                kernel_layer.begin_deep_angles = angle_idx;
                angle_idx += layer.size;
            }
        }
    }
    for(auto layer_idx = 0u; layer_idx < this->num_layers; layer_idx++) {
        auto& layer = kernel::PsiDeepT<dtype, symmetric>::layers[layer_idx];
        auto next_layer = kernel::PsiDeepT<dtype, symmetric>::layers + layer_idx + 1;

        layer.rhs_connectivity = (
            layer_idx + 1 < this->num_layers ?
            next_layer->size * next_layer->lhs_connectivity / layer.size :
            0u
        );
    }
    this->num_final_weights = this->layers.back().size;
    this->num_params += this->num_final_weights;

    this->update_kernel();
}


template<typename dtype, bool symmetric>
void PsiDeepT<dtype, symmetric>::update_kernel() {
    for(auto layer_idx = 0u; layer_idx < this->num_layers; layer_idx++) {
        Layer& layer = *next(this->layers.begin(), layer_idx);
        auto& kernel_layer = this->kernel().layers[layer_idx];

        layer.lhs_connections.update_device();
        layer.rhs_connections.update_device();
        layer.lhs_weights.update_device();
        layer.rhs_weights.update_device();
        layer.biases.update_device();

        kernel_layer.lhs_connections = layer.lhs_connections.data();
        kernel_layer.rhs_connections = layer.rhs_connections.data();
        kernel_layer.lhs_weights = layer.lhs_weights.data();
        kernel_layer.rhs_weights = layer.rhs_weights.data();
        kernel_layer.biases = layer.biases.data();
    }

    this->input_weights.update_device();
    this->final_weights.update_device();

    this->kernel().input_weights = this->input_weights.data();
    this->kernel().final_weights = this->final_weights.data();
}


template<typename dtype, bool symmetric>
pair<Array<unsigned int>, Array<dtype>> PsiDeepT<dtype, symmetric>::compile_rhs_connections_and_weights(
    const unsigned int prev_size,
    const unsigned int size,
    const unsigned int lhs_connectivity,
    const Array<unsigned int>& lhs_connections,
    const Array<dtype>& lhs_weights
) {
    const auto rhs_connectivity = size * lhs_connectivity / prev_size;

    Array<unsigned int> rhs_connections(prev_size * rhs_connectivity, this->gpu);
    Array<dtype> rhs_weights(prev_size * rhs_connectivity, this->gpu);

    vector<unsigned int> lhs_num_connections;
    lhs_num_connections.assign(prev_size, 0u);

    for(auto j = 0u; j < size; j++) {
        for(auto i = 0u; i < lhs_connectivity; i++) {
            const auto lhs_idx = lhs_connections[i * size + j];

            rhs_connections[lhs_idx * rhs_connectivity + lhs_num_connections[lhs_idx]] = j;
            rhs_weights[lhs_idx * rhs_connectivity + lhs_num_connections[lhs_idx]] = lhs_weights[
                i * size + j
            ];
            lhs_num_connections[lhs_idx]++;
        }
    }

    return {move(rhs_connections), move(rhs_weights)};
}


template<typename dtype, bool symmetric>
Array<dtype> PsiDeepT<dtype, symmetric>::get_params() const {
    Array<dtype> result(this->num_params, false);

    for(auto i = 0u; i < this->N; i++) {
        result[i] = this->input_weights[i];
    }
    auto it = result.begin() + this->N;

    for(auto layer_it = next(this->layers.begin()); layer_it != this->layers.end(); layer_it++) {
        auto& layer = *layer_it;

        copy(layer.biases.begin(), layer.biases.end(), it);
        it += layer.biases.size();
        copy(layer.lhs_weights.begin(), layer.lhs_weights.end(), it);
        it += layer.lhs_weights.size();
    }
    copy(this->final_weights.begin(), this->final_weights.end(), it);
    it += this->num_final_weights;

    return result;
}


template<typename dtype, bool symmetric>
void PsiDeepT<dtype, symmetric>::set_params(const Array<dtype>& new_params) {
    for(auto i = 0u; i < this->N; i++) {
        this->input_weights[i] = new_params[i];
    }
    auto it = new_params.begin() + this->N;

    for(auto layer_it = next(this->layers.begin()); layer_it != this->layers.end(); layer_it++) {
        auto& layer = *layer_it;

        copy(it, it + layer.biases.size(), layer.biases.begin());
        it += layer.size;

        copy(it, it + layer.lhs_weights.size(), layer.lhs_weights.begin());
        it += layer.lhs_weights.size();

        prev(layer_it)->rhs_weights = this->compile_rhs_connections_and_weights(
            prev(layer_it)->size,
            layer.size,
            layer.lhs_connectivity,
            layer.lhs_connections,
            layer.lhs_weights
        ).second;
    }
    copy(it, it + this->num_final_weights, this->final_weights.begin());
    it += this->num_final_weights;

    this->update_kernel();
}


// template struct PsiDeepT<cuda_complex::complex<float>>;

#ifdef PSI_DEEP_SYMMETRIC
template struct PsiDeepT<cuda_complex::complex<double>, true>;
#else
template struct PsiDeepT<cuda_complex::complex<double>, false>;
#endif // PSI_DEEP_SYMMETRIC

} // namespace ann_on_gpu
