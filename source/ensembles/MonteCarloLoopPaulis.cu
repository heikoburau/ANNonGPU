#ifdef ENABLE_MONTE_CARLO_PAULIS

#include "ensembles/MonteCarloLoopPaulis.hpp"
#include "quantum_states.hpp"

#include <cassert>


namespace ann_on_gpu {


MonteCarloLoopPaulis::MonteCarloLoopPaulis(
    const unsigned int num_samples,
    const unsigned int num_sweeps,
    const unsigned int num_thermalization_sweeps,
    const unsigned int num_markov_chains,
    const Operator     update_operator
) : gpu(update_operator.gpu),
    acceptances_ar(1, gpu),
    rejections_ar(1, gpu),
    rng_states(num_markov_chains, gpu),
    update_operator_host(update_operator) {

    this->num_samples = num_samples;
    this->num_sweeps = num_sweeps;
    this->num_thermalization_sweeps = num_thermalization_sweeps;
    this->num_markov_chains = num_markov_chains;

    this->weight = 1.0 / num_samples;

    this->num_mc_steps_per_chain = this->num_samples / this->num_markov_chains;

    this->acceptances = this->acceptances_ar.data();
    this->rejections = this->rejections_ar.data();
    this->update_operator = this->update_operator_host.kernel();

    this->kernel().rng_states = this->rng_states.kernel();
}

MonteCarloLoopPaulis::MonteCarloLoopPaulis(MonteCarloLoopPaulis& other)
    :
    acceptances_ar(1, other.gpu),
    rejections_ar(1, other.gpu),
    rng_states(other.rng_states),
    update_operator_host(other.update_operator_host),
    gpu(other.gpu)
{
    this->kernel() = other.kernel();

    this->acceptances = this->acceptances_ar.data();
    this->rejections = this->rejections_ar.data();
    this->update_operator = this->update_operator_host.kernel();

    this->kernel().rng_states = this->rng_states.kernel();
}

} // namespace ann_on_gpu


#endif  // ENABLE_MONTE_CARLO_PAULIS
