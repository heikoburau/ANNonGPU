from ._pyANNonGPU import PsiCNN, log_psi
from .json_numpy import NumpyEncoder, NumpyDecoder
import json
import numpy as np


def to_json(self):
    obj = dict(
        type="PsiCNN",
        extent=self.extent,
        num_channels_list=self.num_channels_list,
        connectivity_list=self.connectivity_list,
        symmetry_classes=self.symmetry_classes,
        params=self.params,
        final_factor=self.final_factor,
        log_prefactor_re=self.log_prefactor.real,
        log_prefactor_im=self.log_prefactor.imag
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
        obj["extent"],
        obj["num_channels_list"],
        obj["connectivity_list"],
        obj["symmetry_classes"],
        obj["params"],
        obj["final_factor"],
        obj["log_prefactor_re"] + 1j * obj["log_prefactor_im"],
        gpu
    )


def normalize(self, exact_summation):
    self.log_prefactor -= np.log(self.norm(exact_summation))


def calibrate(self, ensemble):
    if ensemble.__class__.__name__.startswith("ExactSummation"):
        self.normalize(ensemble)
        self.log_prefactor -= 1j * log_psi(self, ensemble).imag
    else:
        self.log_prefactor = 0
        self.log_prefactor -= log_psi(self, ensemble)


def prod(x_list):
    r = 1
    for x in x_list:
        r *= x
    return r


def channel_link(self, layer, i, j):
    num_channels_list = self.num_channels_list
    connectivity_list = self.connectivity_list
    num_symmetry_classes = self.num_symmetry_classes

    offset = 0
    for l in range(layer):
        num_channel_links = num_channels_list[l] * (num_channels_list[l - 1] if l > 0 else 1)

        offset += num_channel_links * num_symmetry_classes * prod(connectivity_list[l])

    num_channels = num_channels_list[layer]
    connectivity = prod(connectivity_list[layer])

    offset += (i * num_channels + j) * num_symmetry_classes * connectivity

    return self.params[offset:offset + num_symmetry_classes * connectivity]


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
