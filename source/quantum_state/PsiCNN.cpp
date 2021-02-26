#ifdef ENABLE_PSI_CNN

#include "quantum_state/PsiCNN.hpp"


namespace ann_on_gpu {


template<typename dtype>
void PsiCNN_t<dtype>::init_kernel() {
    this->num_layers = this->num_channels_list.size();
    this->num_params = this->params.size();

    this->num_angles = 0u;
    auto params_offset = 0u;
    for(auto l = 0u; l < this->num_layers; l++) {
        auto& layer = this->layers[l];

        layer.num_channels = this->num_channels_list[l];
        layer.num_channel_links = layer.num_channels * (
            l > 0 ? this->layers[l - 1u].num_channels : 1u
        );
        layer.connectivity = this->connectivity_list[l];

        for(auto cl = 0u; cl < layer.num_channel_links; cl++) {
            auto& channel_link = layer.channel_links[cl];

            channel_link.begin_params = params_offset;
            channel_link.weights = this->params.data() + params_offset;

            params_offset += layer.connectivity * this->N;
        }

        this->num_angles += layer.num_channels * this->N;
    }
}


template<typename dtype>
template<typename Ensemble>
void PsiCNN_t<dtype>::init(const Ensemble& ensemble) {
    this->angles.resize(ensemble.get_num_steps() * this->num_angles);

    auto angles_offset = 0u;
    for(auto l = 0u; l < this->num_layers; l++) {
        auto& layer = this->layers[l];

        for(auto c = 0u; c < layer.num_channels; c++) {
            auto& channel = layer.channels[c];

            channel.angles = this->angles.data() + angles_offset;

            angles_offset += this->N;
        }
    }
}


template struct PsiCNN_t<double>;
template struct PsiCNN_t<complex_t>;


} // namespace ann_on_gpu

#endif // ENABLE_PSI_CNN
