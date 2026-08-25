"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _upper_llr,
)
from encoder import bit_reversal_permutation


def _crc_poly_bits(crc_length):
    if crc_length == 8:
        return np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int8)
    if crc_length == 16:
        return np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=np.int8)
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly_bits(crc_length)
    msg = np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)])
    for i in range(len(info_bits)):
        if msg[i] == 1:
            msg[i:i + len(poly)] ^= poly
    crc_bits = msg[len(info_bits):]
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly_bits(crc_length)
    msg = bits.copy()
    for i in range(len(bits) - crc_length + 1):
        if msg[i] == 1:
            end = min(i + len(poly), len(msg))
            msg[i:end] ^= poly[: end - i]
    return np.all(msg[-crc_length:] == 0)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _path_metric_update(self, pm, llr, u):
        if (u == 0 and llr >= 0) or (u == 1 and llr < 0):
            return pm
        return pm + abs(llr)

    def _update_llrs(self, paths, l):
        n = self.n
        N = self.N
        for path in paths:
            L, B = path["L"], path["B"]
            for s in range(n - _active_llr_level(l, n), n):
                block_size = 1 << (s + 1)
                branch_size = block_size // 2
                for j in range(l, N, block_size):
                    if j % block_size < branch_size:
                        L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                    else:
                        L[j, s + 1] = _lower_llr(
                            L[j, s],
                            L[j - branch_size, s],
                            B[j - branch_size, s + 1],
                        )

    def _update_bits(self, paths, l):
        n = self.n
        N = self.N
        if l < N // 2:
            return
        for path in paths:
            B = path["B"]
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N, L = self.n, self.N, self.list_size

        paths = [{
            "pm": 0.0,
            "L": np.zeros((N, n + 1), dtype=np.float64),
            "B": np.zeros((N, n + 1), dtype=np.int8),
        }]
        paths[0]["L"][:, 0] = llr_ch[self.br]

        for phase in range(N):
            l = _bit_reversed(phase, n)
            self._update_llrs(paths, l)

            candidates = []
            for path in paths:
                llr = path["L"][l, n]
                if self.frozen_bits[l]:
                    pm = self._path_metric_update(path["pm"], llr, 0)
                    child = {
                        "pm": pm,
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                    }
                    child["B"][l, n] = 0
                    candidates.append(child)
                else:
                    for u in (0, 1):
                        pm = self._path_metric_update(path["pm"], llr, u)
                        child = {
                            "pm": pm,
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                        }
                        child["B"][l, n] = u
                        candidates.append(child)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[:L]
            self._update_bits(paths, l)

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            valid = []
            for p in paths:
                info_bits = p["B"][info_idx, n]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["B"][:, n].astype(int), best["pm"]
