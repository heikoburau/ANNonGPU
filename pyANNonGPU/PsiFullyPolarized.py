from ._pyANNonGPU import PsiFullyPolarized


def to_json(self):
    return dict(
        type="PsiFullyPolarized",
        num_sites=self.num_sites,
        log_prefactor_re=self.log_prefactor.real,
        log_prefactor_im=self.log_prefactor.imag
    )


@staticmethod
def from_json(obj):
    return PsiFullyPolarized(
        obj["num_sites"],
        obj["log_prefactor_re"] + 1j * obj["log_prefactor_im"]
    )


setattr(PsiFullyPolarized, "to_json", to_json)
setattr(PsiFullyPolarized, "from_json", from_json)
