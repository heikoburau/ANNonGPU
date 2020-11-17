#pragma once

#include "types.h"

#ifdef __CUDACC__
    #include "curand_kernel.h"
#else
    struct curandState_t {};
#endif // __CUDACC__
#include <random>


namespace ann_on_gpu {

using namespace std;


namespace kernel {


struct RNGStates {
    curandState_t*  rng_states_device;
    mt19937*        rng_states_host;

#ifdef __CUDACC__

    template<typename RNGState_t>
    HDINLINE void get_state(RNGState_t& result, const unsigned int idx) const {
        #ifdef __CUDA_ARCH__
        result = this->rng_states_device[idx];
        #else
        result = this->rng_states_host[idx];
        #endif
    }

    HDINLINE void get_state(void* result, const unsigned int idx) const {
        #ifdef __CUDA_ARCH__
        *reinterpret_cast<curandState_t*>(result) = this->rng_states_device[idx];
        #else
        *reinterpret_cast<mt19937*>(result) = this->rng_states_host[idx];
        #endif
    }

    template<typename RNGState_t>
    HDINLINE void set_state(const RNGState_t& new_state, const unsigned int idx) const {
        #ifdef __CUDA_ARCH__
        this->rng_states_device[idx] = new_state;
        #else
        this->rng_states_host[idx] = new_state;
        #endif
    }

    HDINLINE void set_state(const void* new_state, const unsigned int idx) const {
        #ifdef __CUDA_ARCH__
        this->rng_states_device[idx] = *reinterpret_cast<const curandState_t*>(new_state);
        #else
        this->rng_states_host[idx] = *reinterpret_cast<const mt19937*>(new_state);
        #endif
    }

#endif // __CUDACC__

    inline RNGStates& kernel() {
        return *this;
    }
};


}  // namespace kernel


struct RNGStates : public kernel::RNGStates {
    bool            gpu;
    unsigned int    num_states;

    RNGStates(const unsigned int num_states, const bool gpu);
    RNGStates(const RNGStates& other);
    ~RNGStates() noexcept(false);

    // void skip_ahead(const unsigned int steps);
};

}  // namespace ann_on_gpu
