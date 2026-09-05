"""
极化码 SCL（串行抵消列表）译码器
"""
import math
import sys
import types
import importlib.util
import numpy as np

from encoder import bit_reversed


def _load_ref_scd():
    if "polarcodes.SCD" in sys.modules:
        return sys.modules["polarcodes.SCD"].SCD
    pkg = types.ModuleType("polarcodes")
    sys.modules["polarcodes"] = pkg

    def _load(name, filename):
        path = f"/workspace/polar_codes/mcba1n/{filename}"
        spec = importlib.util.spec_from_file_location(f"polarcodes.{name}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"polarcodes.{name}"] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("utils", "utils_ref.py")
    _load("decoder_utils", "decoder_utils_ref.py")
    scd_mod = _load("SCD", "scd_ref.py")
    return scd_mod.SCD


_SCD = _load_ref_scd()


CRC_POLYNOMIALS = {8: 0x07, 16: 0x8005}


def _crc_remainder(bits, crc_length=8):
    poly = CRC_POLYNOMIALS[crc_length]
    g = (1 << crc_length) | poly
    data = np.asarray(bits, dtype=np.int8).astype(int).tolist()
    for i in range(len(data) - crc_length):
        if data[i]:
            for j in range(crc_length + 1):
                if i + j < len(data):
                    data[i + j] ^= (g >> (crc_length - j)) & 1
    return np.array(data[-crc_length:], dtype=np.int8)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    remainder = _crc_remainder(
        np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)]),
        crc_length,
    )
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) <= crc_length:
        return False
    return bool(np.all(_crc_remainder(bits, crc_length) == 0))


def _frozen_indices(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return np.where(fb)[0]
    return np.where(fb != 0)[0]


class _Path:
    __slots__ = ("scd", "pm", "u_hat", "N", "n", "frozen")

    def __init__(self, llr_ch, frozen_idx, N, n):
        self.N = N
        self.n = n
        self.frozen = frozen_idx
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

        class _PC:
            pass

        pc = _PC()
        pc.N = N
        pc.n = n
        pc.frozen = frozen_idx
        pc.likelihoods = llr_ch.copy()
        self.scd = _SCD(pc)

    def copy(self):
        p = _Path.__new__(_Path)
        p.N, p.n, p.frozen = self.N, self.n, self.frozen
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()

        class _PC:
            pass

        pc = _PC()
        pc.N = self.N
        pc.n = self.n
        pc.frozen = self.frozen
        pc.likelihoods = self.scd.L[:, 0].copy()
        p.scd = _SCD(pc)
        p.scd.L = self.scd.L.copy()
        p.scd.B = self.scd.B.copy()
        return p


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_idx = _frozen_indices(frozen_bits)
        self.frozen_set = set(self.frozen_idx)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.array(sorted(set(range(N)) - self.frozen_set))

    def _pm_add(self, pm, llr_val, bit):
        hard = 0 if llr_val >= 0 else 1
        return pm + (0.0 if bit == hard else abs(llr_val))

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            from decoder_sc import sc_decode

            frozen_bits = np.ones(self.N, dtype=int)
            frozen_bits[self.info_indices] = 0
            return sc_decode(llr_ch, frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(llr_ch, self.frozen_idx, self.N, self.n)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                path.scd.update_llrs(l)
                llr_val = path.scd.L[l, self.n]

                if l in self.frozen_set:
                    p = path.copy()
                    p.pm = self._pm_add(p.pm, llr_val, 0)
                    p.u_hat[l] = 0
                    p.scd.B[l, self.n] = 0
                    p.scd.update_bits(l)
                    new_paths.append(p)
                else:
                    for bit in (0, 1):
                        p = path.copy()
                        p.pm = self._pm_add(p.pm, llr_val, bit)
                        p.u_hat[l] = bit
                        p.scd.B[l, self.n] = bit
                        p.scd.update_bits(l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = min(valid or paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
