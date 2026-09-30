"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)


CRC_POLYS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    top = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    if crc_length not in CRC_POLYS:
        raise ValueError("crc_length must be 8 or 16")
    poly = CRC_POLYS[crc_length]
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([np.asarray(info_bits, dtype=int), crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    if crc_length not in CRC_POLYS:
        raise ValueError("crc_length must be 8 or 16")
    poly = CRC_POLYS[crc_length]
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


class _Path:
    __slots__ = ("L", "B", "pm")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0

    def copy(self):
        p = _Path.__new__(_Path)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        return p


class SCLDecoder:
    """SCL 译码器（路径复制实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _path_metric_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[br]

        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr_leaf = path.L[l, self.n]

                if l in self.frozen_set:
                    u0 = path.copy()
                    u0.pm += self._path_metric_penalty(llr_leaf, 0)
                    u0.B[l, self.n] = 0
                    _update_bits(u0.B, l, self.n)
                    new_paths.append(u0)
                else:
                    for u_bit in (0, 1):
                        pb = path.copy()
                        pb.pm += self._path_metric_penalty(llr_leaf, u_bit)
                        pb.B[l, self.n] = u_bit
                        _update_bits(pb.B, l, self.n)
                        new_paths.append(pb)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        candidates = []
        for p in paths:
            u_hat = p.B[:, self.n].astype(int)
            if self.crc_length > 0:
                payload = u_hat[self.info_indices]
                if not crc_check(payload, self.crc_length):
                    continue
            candidates.append((p.pm, u_hat))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1], candidates[0][0]

        best = min(paths, key=lambda p: p.pm)
        return best.B[:, self.n].astype(int), best.pm
