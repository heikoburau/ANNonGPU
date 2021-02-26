from ._pyANNonGPU import PsiCNN, log_psi
from .json_numpy import NumpyEncoder, NumpyDecoder
import json


def to_json(self):
    obj = dict(
        type="PsiCNN",
        num_sites=self.num_sites,
        N=self.N,
        num_channels_list=self.num_channels_list,
        connectivity_list=self.connectivity_list,
        params=self.params,
        final_factor=self.final_factor,
        prefactor=self.prefactor
    )

    return json.loads(
        json.dumps(obj, cls=NumpyEncoder)
    )


@staticmethod
def from_json(json_obj, gpu):
    obj = json.loads(
        json.dumps(
            json_obj,
            cls=NumpyEncoder
        ),
        cls=NumpyDecoder
    )

    return PsiCNN(
        obj["num_sites"],
        obj["N"],
        obj["num_channels_list"],
        obj["connectivity_list"],
        obj["params"],
        obj["final_factor"],
        obj["prefactor"],
        gpu
    )


def normalize(self, exact_summation):
    self.prefactor = 1
    self.prefactor /= self.norm(exact_summation)


def calibrate(self, ensemble):
    if ensemble.__class__.__name__.startswith("ExactSummation"):
        self.prefactor = 1
        self.log_prefactor = 0
        self.prefactor /= self.norm(ensemble)
        self.log_prefactor -= log_psi(self, ensemble)
        self.prefactor /= self.norm(ensemble)
    else:
        self.log_prefactor = 0
        self.log_prefactor -= log_psi(self, ensemble)


def channel_link(self, layer, i, j):
    num_channels_list = self.num_channels_list
    connectivity_list = self.connectivity_list

    offset = 0
    for l in range(layer):
        num_channel_links = num_channels_list[l] * (num_channels_list[l - 1] if l > 0 else 1)

        offset += num_channel_links * connectivity_list[l]

    num_channels = num_channels_list[layer]
    connectivity = connectivity_list[layer]

    offset += (i * num_channels + j) * connectivity

    return self.params[offset:offset + connectivity]


def __pos__(self):
    return self.copy()


@property
def vector(self):
    return self._vector


setattr(PsiCNN, "to_json", to_json)
setattr(PsiCNN, "from_json", from_json)
setattr(PsiCNN, "normalize", normalize)
setattr(PsiCNN, "calibrate", calibrate)
setattr(PsiCNN, "channel_link", channel_link)
setattr(PsiCNN, "__pos__", __pos__)
setattr(PsiCNN, "vector", vector)
