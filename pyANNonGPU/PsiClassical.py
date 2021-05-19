from . import _pyANNonGPU
from .json_numpy import NumpyEncoder, NumpyDecoder
from QuantumExpression import PauliExpression
import json


def vCN_to_json(self):
    psi_ref = self.psi_ref
    if hasattr(psi_ref, "to_json"):
        psi_ref = psi_ref.to_json()

    obj = dict(
        type="PsiClassical",
        num_sites=self.num_sites,
        order=self.order,
        H_local=[h.expr.to_json() for h in self.H_local],
        params=self.params[:len(self.H_local)],
        psi_ref=psi_ref,
        log_prefactor_re=self.log_prefactor.real,
        log_prefactor_im=self.log_prefactor.imag
    )

    return json.loads(
        json.dumps(obj, cls=NumpyEncoder)
    )


def vCN_from_json(json_obj, gpu):
    obj = json.loads(
        json.dumps(
            json_obj,
            cls=NumpyEncoder
        ),
        cls=NumpyDecoder
    )

    if obj["psi_ref"]["type"] == "PsiFullyPolarized":
        psi_ref = _pyANNonGPU.PsiFullyPolarized.from_json(obj["psi_ref"])

    if obj["psi_ref"]["type"] == "PsiDeep":
        psi_ref = _pyANNonGPU.PsiDeep.from_json(obj["psi_ref"], gpu)

    H_local = [
        _pyANNonGPU.Operator(PauliExpression.from_json(h), gpu)
        for h in obj["H_local"]
    ]

    log_prefactor = obj["log_prefactor_re"] + 1j * obj["log_prefactor_im"]

    if obj["order"] == 1:
        if obj["psi_ref"]["type"] == "PsiFullyPolarized":
            return _pyANNonGPU.PsiClassicalFP_1(
                obj["num_sites"], H_local, obj["params"], psi_ref, log_prefactor, gpu
            )
        else:
            return _pyANNonGPU.PsiClassicalANN_1(
                obj["num_sites"], H_local, obj["params"], psi_ref, log_prefactor, gpu
            )
    elif obj["order"] == 2:
        if obj["psi_ref"]["type"] == "PsiFullyPolarized":
            return _pyANNonGPU.PsiClassicalFP_2(
                obj["num_sites"], H_local, obj["params"], psi_ref, log_prefactor, gpu
            )
        else:
            return _pyANNonGPU.PsiClassicalANN_2(
                obj["num_sites"], H_local, obj["params"], psi_ref, log_prefactor, gpu
            )
