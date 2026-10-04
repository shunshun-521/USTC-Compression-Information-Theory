"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助 CA-SCL
"""
import math
import numpy as np
from decoder_sc import _minsum, f_operation, g_operation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    poly = _crc_poly(crc_length)
    reg = 0
    for b in info_bits:
        reg <<= 1
        reg |= int(b)
        if reg & (1 << crc_length):
            reg ^= poly
    reg &= (1 << crc_length) - 1
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([np.asarray(info_bits, dtype=int), np.asarray(crc_bits, dtype=int)])


def crc_check(bits, crc_length=8):
    poly = _crc_poly(crc_length)
    reg = 0
    for b in bits:
        reg <<= 1
        reg |= int(b)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg == 0


class _Path:
    __slots__ = ("pm", "R", "u_hat", "active")

    def __init__(self, m, N):
        self.pm = 0.0
        self.R = np.zeros((N, m + 1), dtype=np.float64)
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.active = True


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.m = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length
        self.info_indices = None if info_indices is None else np.asarray(info_indices, dtype=int)

    def _llr_for_path(self, L, R, phi, llr_ch):
        L[:, self.m] = llr_ch
        for j in range(self.m, 0, -1):
            s = 1 << (j - 1)
            for i in range(0, self.N, s << 1):
                for k in range(s):
                    idx = i + k
                    idx2 = idx + s
                    L[idx, j - 1] = _minsum(R[idx, j] + L[idx2, j], L[idx, j])
                    L[idx2, j - 1] = _minsum(R[idx, j], L[idx, j]) + L[idx2, j]
        return L[phi, 0] + R[phi, 0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        large = 1e8
        paths = [_Path(self.m, self.N)]
        paths[0].R[self.frozen_bits, 0] = large
        L_store = [np.zeros((self.N, self.m + 1), dtype=np.float64)]

        for phi in range(self.N):
            candidates = []
            for pi, path in enumerate(paths):
                if not path.active:
                    continue
                L = L_store[pi]
                llr_dec = self._llr_for_path(L, path.R, phi, llr_ch)
                if self.frozen_bits[phi]:
                    pen = 0.0 if llr_dec >= 0 else abs(llr_dec)
                    candidates.append((path.pm + pen, pi, 0, path))
                else:
                    pen0 = 0.0 if llr_dec >= 0 else abs(llr_dec)
                    pen1 = 0.0 if llr_dec < 0 else abs(llr_dec)
                    candidates.append((path.pm + pen0, pi, 0, path))
                    candidates.append((path.pm + pen1, pi, 1, path))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.L]

            new_paths = []
            new_L_store = []
            for pm, pidx, ubit, _ in candidates:
                parent = paths[pidx]
                child = _Path(self.m, self.N)
                child.pm = pm
                child.R = parent.R.copy()
                child.u_hat = parent.u_hat.copy()
                child.u_hat[phi] = ubit
                child.R[phi, 0] = large if ubit == 0 else -large
                new_paths.append(child)
                new_L_store.append(L_store[pidx].copy())
            paths = new_paths
            L_store = new_L_store

        best = min(paths, key=lambda p: p.pm)
        if self.crc_length > 0:
            valid = []
            for p in paths:
                if self.info_indices is None:
                    continue
                bits = p.u_hat[self.info_indices]
                if crc_check(bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p.pm)
        return best.u_hat.astype(int), float(best.pm)
