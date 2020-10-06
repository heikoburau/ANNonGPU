#ifdef ENABLE_MONTE_CARLO

#include "ensembles/MonteCarloLoop.hpp"
#include "quantum_states.hpp"

#include <cassert>


namespace ann_on_gpu {

__global__ void kernel_initialize_random_states(curandState_t* random_states, const unsigned int num_markov_chains) {
    const auto markov_index = blockIdx.x * blockDim.x + threadIdx.x;

    if(markov_index < num_markov_chains) {
        curand_init(0, markov_index, 0u, &random_states[markov_index]);
    }
}


MonteCarloLoop::MonteCarloLoop(
    const unsigned int num_samples,
    const unsigned int num_sweeps,
    const unsigned int num_thermalization_sweeps,
    const unsigned int num_markov_chains,
    const bool         gpu
) : acceptances_ar(1, gpu), rejections_ar(1, gpu), rng_states(num_markov_chains, gpu), gpu(gpu) {
    this->num_samples = num_samples;
    this->num_sweeps = num_sweeps;
    this->num_thermalization_sweeps = num_thermalization_sweeps;
    this->num_markov_chains = num_markov_chains;
    this->has_total_z_symmetry = false;

    this->weight = 1.0 / num_samples;

    this->num_mc_steps_per_chain = this->num_samples / this->num_markov_chains;

    this->fast_sweep = false;
    this->fast_sweep_num_tries = 0u;

    this->acceptances = this->acceptances_ar.data();
    this->rejections = this->rejections_ar.data();

    this->kernel().rng_states = this->rng_states.kernel();
}

MonteCarloLoop::MonteCarloLoop(MonteCarloLoop& other)
    :
    acceptances_ar(1, other.gpu),
    rejections_ar(1, other.gpu),
    rng_states(other.rng_states),
    gpu(other.gpu)
{
    this->num_samples = other.num_samples;
    this->num_sweeps = other.num_sweeps;
    this->num_thermalization_sweeps = other.num_thermalization_sweeps;
    this->num_markov_chains = other.num_markov_chains;
    this->has_total_z_symmetry = other.has_total_z_symmetry;
    this->symmetry_sector = other.symmetry_sector;

    this->weight = other.weight;

    this->num_mc_steps_per_chain = this->num_samples / this->num_markov_chains;

    this->fast_sweep = other.fast_sweep;
    this->fast_sweep_num_tries = other.fast_sweep_num_tries;

    this->acceptances = this->acceptances_ar.data();
    this->rejections = this->rejections_ar.data();

    this->kernel().rng_states = this->rng_states.kernel();
}

} // namespace ann_on_gpu


#endif  // ENABLE_MONTE_CARLO
