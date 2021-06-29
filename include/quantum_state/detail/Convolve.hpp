#pragma once


#include "types.h"


namespace ann_on_gpu {
namespace detail {


template<unsigned int dim>
struct Convolve;


template<>
struct Convolve<1u> {

    template<typename Function>
    HDINLINE void foreach_connection(
        const unsigned int idx,
        const unsigned int extent[1u],
        const unsigned int connectivity[1u],
        Function function
    ) const {
        for(auto i = 0u; i < connectivity[0]; i++) {
            function(i, (idx + i) % extent[0]);
        }
    }

    template<typename dtype>
    HDINLINE dtype operator()(
        const unsigned int idx,
        const unsigned int extent[1u],
        const unsigned int connectivity[1u],
        dtype* weights,
        dtype* input_activations
    ) const {
        dtype result(0.0);

        this->foreach_connection(
            idx, extent, connectivity,
            [&](const unsigned int conn_idx, const unsigned int input_idx){
                result += weights[conn_idx] * input_activations[input_idx];
            }
        );

        return result;
    }
};


template<>
struct Convolve<2u> {

    template<typename Function>
    HDINLINE void foreach_connection(
        const unsigned int idx,
        const unsigned int extent[2u],
        const unsigned int connectivity[2u],
        Function function
    ) const {
        const auto row = idx / extent[1];
        const auto col = idx % extent[1];

        for(auto i = 0u; i < connectivity[0]; i++) {
            for(auto j = 0u; j < connectivity[1]; j++) {
                function(
                    i * connectivity[1] + j,
                    (
                        ((row + i) % extent[0]) * extent[1] +
                        ((col + j) % extent[1])
                    )
                );
            }
        }
    }

    template<typename dtype>
    HDINLINE dtype operator()(
        const unsigned int idx,
        const unsigned int extent[2u],
        const unsigned int connectivity[2u],
        dtype* weights,
        dtype* input_activations
    ) const {
        dtype result(0.0);

        this->foreach_connection(
            idx, extent, connectivity,
            [&](const unsigned int conn_idx, const unsigned int input_idx){
                result += weights[conn_idx] * input_activations[input_idx];
            }
        );

        return result;
    }
};

template<>
struct Convolve<3u> {

    template<typename Function>
    HDINLINE void foreach_connection(
        const unsigned int idx,
        const unsigned int extent[3u],
        const unsigned int connectivity[3u],
        Function function
    ) const {
        const auto page_size = extent[1] * extent[2];
        const auto page = idx / page_size;
        const auto row = (idx % page_size) / extent[2];
        const auto col = (idx % page_size) % extent[2];

        auto c_idx = 0u;
        for(auto k = 0u; k < connectivity[0]; k++) {
            for(auto i = 0u; i < connectivity[1]; i++) {
                for(auto j = 0u; j < connectivity[2]; j++) {
                    function(
                        c_idx,
                        (
                            ((page + k) % extent[0]) * page_size +
                            ((row + i) % extent[1]) * extent[2] +
                            ((col + j) % extent[2])
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
        const unsigned int extent[3u],
        const unsigned int connectivity[3u],
        dtype* weights,
        dtype* input_activations
    ) const {
        dtype result(0.0);

        this->foreach_connection(
            idx, extent, connectivity,
            [&](const unsigned int conn_idx, const unsigned int input_idx){
                result += weights[conn_idx] * input_activations[input_idx];
            }
        );

        return result;
    }

};

} // namespace detail
} // namespace ann_on_gpu
