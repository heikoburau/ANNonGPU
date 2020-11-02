from ._pyANNonGPU import PsiDeep
from .json_numpy import NumpyEncoder, NumpyDecoder
import json


def to_json(self):
    obj = dict(
        type="PsiDeep",
        num_sites=self.num_sites,
        a=self.a,
        b=self.b,
        connections=self.connections,
        W=self.W,
        final_weights=self.final_weights,
        translational_invariance=self.translational_invariance,
        prefactor=self.prefactor,
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

    return PsiDeep(
        obj["num_sites"],
        obj["a"],
        obj["b"],
        obj["connections"],
        obj["W"],
        obj["final_weights"],
        obj["prefactor"],
        obj["translational_invariance"],
        gpu
    )


def normalize(self, exact_summation):
    self.prefactor = 1
    self.prefactor /= self.norm(exact_summation)


def __pos__(self):
    return self.copy()


@property
def vector(self):
    return self._vector


@property
def first_layer_params_slice(self):
    return slice(self.N, self.N + self.W[0].size + self.b[0].size)


setattr(PsiDeep, "to_json", to_json)
setattr(PsiDeep, "from_json", from_json)
setattr(PsiDeep, "normalize", normalize)
setattr(PsiDeep, "__pos__", __pos__)
setattr(PsiDeep, "vector", vector)
setattr(PsiDeep, "first_layer_params_slice", first_layer_params_slice)
