#ifdef ENABLE_MONTE_CARLO

#include "ensembles/MonteCarlo.hpp"
#include "bases.hpp"
#include "quantum_states.hpp"

#include <cassert>


namespace ann_on_gpu {


template<typename Basis_t, typename Init_Policy, typename Update_Policy>
MonteCarlo_t<Basis_t, Init_Policy, Update_Policy>::MonteCarlo_t(
    const unsigned int  num_samples,
    const unsigned int  num_sweeps,
    const unsigned int  num_thermalization_sweeps,
    const unsigned int  num_markov_chains,
    const Update_Policy update_policy,
    const bool          gpu
) :
    acceptances_ar(1, gpu),
    rejections_ar(1, gpu),
    rng_states(num_markov_chains, gpu),
    update_policy(update_policy),
    gpu(gpu) {

    this->num_samples = num_samples;
    this->num_sweeps = num_sweeps;
    this->num_thermalization_sweeps = num_thermalization_sweeps;
    this->num_markov_chains = num_markov_chains;

    this->weight = 1.0 / num_samples;

    this->num_mc_steps_per_chain = this->num_samples / this->num_markov_chains;

    this->acceptances = this->acceptances_ar.data();
    this->rejections = this->rejections_ar.data();

    this->kernel().rng_states = this->rng_states.kernel();
    this->kernel().update_policy = this->update_policy.kernel();
}


template<typename Basis_t, typename Init_Policy, typename Update_Policy>
MonteCarlo_t<Basis_t, Init_Policy, Update_Policy>::MonteCarlo_t(const MonteCarlo_t& other)
    :
    acceptances_ar(other.acceptances_ar),
    rejections_ar(other.rejections_ar),
    rng_states(other.rng_states),
    update_policy(other.update_policy),
    gpu(other.gpu)
{
    this->kernel() = other.kernel();

    this->acceptances = this->acceptances_ar.data();
    this->rejections = this->rejections_ar.data();

    this->kernel().rng_states = this->rng_states.kernel();
    this->kernel().update_policy = this->update_policy.kernel();
}


#ifdef ENABLE_SPINS

template struct MonteCarlo_t<Spins, Init_Policy<Spins>, Update_Policy<Spins>>;

#endif // ENABLE_SPINS


#ifdef ENABLE_PAULIS

template struct MonteCarlo_t<PauliString, Init_Policy<PauliString>, Update_Policy<PauliString>>;

#endif // ENABLE_PAULIS


} // namespace ann_on_gpu


#endif  // ENABLE_MONTE_CARLO
