"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    f_operation_exact,
    g_operation,
)
from encoder import bit_reversal_permutation as br_perm


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    for b in np.asarray(info_bits, dtype=int):
        reg ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits.astype(int), crc_bits])


def crc_check(bits, crc_length=8):
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时共享 L/B，写入前复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.br = br_perm(N)

    def _new_path(self, llr_ch):
        return {
            "pm": 0.0,
            "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=np.int8),
            "owned": True,
        }

    def _ensure_owned(self, path):
        if not path["owned"]:
            path["L"] = path["L"].copy()
            path["B"] = path["B"].copy()
            path["owned"] = True

    def _llr_update(self, path, l):
        L, B = path["L"], path["B"]
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation_exact(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

    def _bit_update(self, path, l):
        L, B = path["L"], path["B"]
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]
        paths[0]["L"][:, 0] = llr_ch

        for phi_nat in range(self.N):
            l = self.br[phi_nat]
            candidates = []

            for path in paths:
                self._ensure_owned(path)
                self._llr_update(path, l)
                llr_bit = path["L"][l, self.n]

                if self.frozen_bits[l]:
                    p = {
                        "pm": path["pm"] + self._penalty(llr_bit, 0),
                        "L": path["L"],
                        "B": path["B"].copy(),
                        "owned": True,
                    }
                    p["B"][l, self.n] = 0
                    self._bit_update(p, l)
                    candidates.append(p)
                else:
                    for u in (0, 1):
                        p = {
                            "pm": path["pm"] + self._penalty(llr_bit, u),
                            "L": path["L"],
                            "B": path["B"].copy(),
                            "owned": True,
                        }
                        p["B"][l, self.n] = u
                        self._bit_update(p, l)
                        candidates.append(p)

            candidates.sort(key=lambda x: x["pm"])
            paths = candidates[: self.list_size]

        best = None
        if self.crc_length > 0:
            for p in sorted(paths, key=lambda x: x["pm"]):
                info_bits = p["B"][:, self.n][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    best = p
                    break
        if best is None:
            best = min(paths, key=lambda p: p["pm"])

        return best["B"][:, self.n].astype(int), best["pm"]
