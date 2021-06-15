from ._pyANNonGPU import PsiRBM, log_psi
from .json_numpy import NumpyEncoder, NumpyDecoder
import json
import numpy as np


def to_json(self):
    obj = dict(
        type="PsiRBM",
        W=self.W,
        final_weight=self.final_weight,
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

    return PsiRBM(
        obj["W"],
        obj["final_weight"],
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


def __pos__(self):
    return self.copy()


@property
def vector(self):
    return self._vector


setattr(PsiRBM, "to_json", to_json)
setattr(PsiRBM, "from_json", from_json)
setattr(PsiRBM, "normalize", normalize)
setattr(PsiRBM, "calibrate", calibrate)
setattr(PsiRBM, "__pos__", __pos__)
setattr(PsiRBM, "vector", vector)
