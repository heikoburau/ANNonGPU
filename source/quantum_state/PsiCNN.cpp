#ifdef ENABLE_PSI_CNN

#include "quantum_state/PsiCNN.hpp"


namespace ann_on_gpu {


template<unsigned int dim, typename dtype>
void PsiCNN_t<dim, dtype>::init_kernel() {
    this->num_layers = this->num_channels_list.size();
    this->num_params = this->params.size();

    this->num_angles = 0u;
    auto params_offset = 0u;

    for(auto d = 0u; d < dim; d++) {
        this->convolve.extent[d] = this->extent[d];
    }
    for(auto i = 0u; i < this->N; i++) {
        this->convolve.symmetry_classes[i] = this->symmetry_classes[i];
    }

    for(auto l = 0u; l < this->num_layers; l++) {
        auto& layer = this->layers[l];

        layer.num_channels = this->num_channels_list[l];
        layer.num_channel_links = layer.num_channels * (
            l > 0 ? this->layers[l - 1u].num_channels : 1u
        );

        layer.connectivity_vol = 1u;
        for(auto d = 0u; d < dim; d++) {
            layer.connectivity[d] = this->connectivity_list[l * dim + d];
            layer.connectivity_vol *= layer.connectivity[d];
        }

        for(auto cl = 0u; cl < layer.num_channel_links; cl++) {
            auto& channel_link = layer.channel_links[cl];

            channel_link.begin_params = params_offset;
            channel_link.weights = this->params.data() + params_offset;

            params_offset += this->num_symmetry_classes * layer.connectivity_vol;
        }

        this->num_angles += layer.num_channels * this->N;
    }

    if(this->angles.empty()) {
        this->angles.resize(this->num_angles);
    }

    this->init_kernel_angles();
}


template<unsigned int dim, typename dtype>
void PsiCNN_t<dim, dtype>::init_kernel_angles() {
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

template<unsigned int dim, typename dtype>
void PsiCNN_t<dim, dtype>::init_gradient(const unsigned int num_steps) {
    this->angles.resize(num_steps * this->num_angles);
    this->init_kernel_angles();
}


template struct PsiCNN_t<1u, double>;
template struct PsiCNN_t<1u, complex_t>;
template struct PsiCNN_t<2u, double>;
template struct PsiCNN_t<2u, complex_t>;
template struct PsiCNN_t<3u, double>;
template struct PsiCNN_t<3u, complex_t>;

} // namespace ann_on_gpu

#endif // ENABLE_PSI_CNN
