"""SCD wrapper around vendored reference implementation."""
import importlib.util
from pathlib import Path

import numpy as np

# Load reference SCD once (mcba1n/polar-codes, MIT)
_REF_ROOT = Path(__file__).resolve().parent / "_ref_scd"
if not (_REF_ROOT / "SCD.py").exists():
    raise RuntimeError("Missing vendored SCD reference files")


def _load_official_scd_class():
    import sys

    spec_u = importlib.util.spec_from_file_location(
        "polarcodes.utils", _REF_ROOT / "utils_min.py"
    )
    mod_u = importlib.util.module_from_spec(spec_u)
    sys.modules["polarcodes.utils"] = mod_u
    spec_u.loader.exec_module(mod_u)

    spec_d = importlib.util.spec_from_file_location(
        "polarcodes.decoder_utils", _REF_ROOT / "decoder_utils.py"
    )
    mod_d = importlib.util.module_from_spec(spec_d)
    sys.modules["polarcodes.decoder_utils"] = mod_d
    spec_d.loader.exec_module(mod_d)

    spec_s = importlib.util.spec_from_file_location(
        "polarcodes.SCD", _REF_ROOT / "SCD.py"
    )
    mod_s = importlib.util.module_from_spec(spec_s)
    sys.modules["polarcodes.SCD"] = mod_s
    spec_s.loader.exec_module(mod_s)
    return mod_s.SCD


SCDOfficial = _load_official_scd_class()


class _PC:
    __slots__ = ("N", "n", "likelihoods", "frozen")

    def __init__(self, N, n, likelihoods, frozen):
        self.N = N
        self.n = n
        self.likelihoods = likelihoods
        self.frozen = frozen


class SCD:
    def __init__(self, N, n, llr_ch, frozen_indices):
        self._pc = _PC(N, n, llr_ch, set(frozen_indices))
        self._dec = SCDOfficial(self._pc)

    @property
    def L(self):
        return self._dec.L

    @L.setter
    def L(self, value):
        self._dec.L = value

    @property
    def B(self):
        return self._dec.B

    @B.setter
    def B(self, value):
        self._dec.B = value

    def update_llrs(self, l):
        self._dec.update_llrs(l)

    def update_bits(self, l):
        self._dec.update_bits(l)

    def decode(self):
        return self._dec.decode()


def bit_reversed(i, n):
    result = 0
    for k in range(n):
        if i & (1 << k):
            result |= 1 << (n - 1 - k)
    return result
