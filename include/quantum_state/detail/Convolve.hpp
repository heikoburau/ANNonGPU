#pragma once


#include "types.h"


namespace ann_on_gpu {
namespace detail {


template<unsigned int dim>
struct Convolve;


template<>
struct Convolve<1u> {
    static constexpr auto dim = 1u;
    static constexpr auto max_N = MAX_SPINS;

    unsigned int N;
    unsigned int extent[dim];
    unsigned int symmetry_classes[max_N];


    template<typename Function>
    HDINLINE void foreach_connection(
        const unsigned int idx,
        const unsigned int connectivity[dim],
        Function function
    ) const {
        for(auto i = 0u; i < connectivity[0]; i++) {
            function(i, (idx + i) % this->extent[0]);
        }
    }

    template<typename dtype>
    HDINLINE dtype operator()(
        const unsigned int idx,
        const unsigned int connectivity[dim],
        dtype* weights,
        dtype* input_activations
    ) const {
        dtype result(0.0);

        this->foreach_connection(
            idx, connectivity,
            [&](const unsigned int conn_idx, const unsigned int input_idx){
                result += weights[this->symmetry_classes[idx] * this->N + conn_idx] * input_activations[input_idx];
            }
        );

        return result;
    }
};


template<>
struct Convolve<2u> {
    static constexpr auto dim = 2u;
    static constexpr auto max_N = MAX_SPINS;

    unsigned int N;
    unsigned int extent[dim];
    unsigned int symmetry_classes[max_N];


    template<typename Function>
    HDINLINE void foreach_connection(
        const unsigned int idx,
        const unsigned int connectivity[dim],
        Function function
    ) const {
        const auto row = idx / this->extent[1];
        const auto col = idx % this->extent[1];

        for(auto i = 0u; i < connectivity[0]; i++) {
            for(auto j = 0u; j < connectivity[1]; j++) {
                function(
                    i * connectivity[1] + j,
                    (
                        ((row + i) % this->extent[0]) * this->extent[1] +
                        ((col + j) % this->extent[1])
                    )
                );
            }
        }
    }

    template<typename dtype>
    HDINLINE dtype operator()(
        const unsigned int idx,
        const unsigned int connectivity[dim],
        dtype* weights,
        dtype* input_activations
    ) const {
        dtype result(0.0);

        this->foreach_connection(
            idx, connectivity,
            [&](const unsigned int conn_idx, const unsigned int input_idx){
                result += weights[this->symmetry_classes[idx] * this->N + conn_idx] * input_activations[input_idx];
            }
        );

        return result;
    }
};

template<>
struct Convolve<3u> {
    static constexpr auto dim = 3u;
    static constexpr auto max_N = MAX_SPINS;

    unsigned int N;
    unsigned int extent[dim];
    unsigned int symmetry_classes[max_N];


    template<typename Function>
    HDINLINE void foreach_connection(
        const unsigned int idx,
        const unsigned int connectivity[dim],
        Function function
    ) const {
        const auto page_size = this->extent[1] * this->extent[2];
        const auto page = idx / page_size;
        const auto row = (idx % page_size) / this->extent[2];
        const auto col = (idx % page_size) % this->extent[2];

        auto c_idx = 0u;
        for(auto k = 0u; k < connectivity[0]; k++) {
            for(auto i = 0u; i < connectivity[1]; i++) {
                for(auto j = 0u; j < connectivity[2]; j++) {
                    function(
                        c_idx,
                        (
                            ((page + k) % this->extent[0]) * page_size +
                            ((row + i) % this->extent[1]) * this->extent[2] +
                            ((col + j) % this->extent[2])
                        )
                    );
                    c_idx++;
                }
            }
        }
    }

    template<typename dtype>
    HDINLINE dtype operator()(
        const unsigned int idx,
        const unsigned int connectivity[dim],
        dtype* weights,
        dtype* input_activations
    ) const {
        dtype result(0.0);

        this->foreach_connection(
            idx, connectivity,
            [&](const unsigned int conn_idx, const unsigned int input_idx){
                result += weights[this->symmetry_classes[idx] * this->N + conn_idx] * input_activations[input_idx];
            }
        );

        return result;
    }

};

} // namespace detail
} // namespace ann_on_gpu
