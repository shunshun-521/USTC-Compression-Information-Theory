"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
    _update_llrs_nonrecursive,
    _update_bits_nonrecursive,
)


# ==================== CRC 工具 ====================

_CRC_POLYS = {8: 0x07, 16: 0x8005}


def _crc_remainder(bits, crc_length):
    poly = _CRC_POLYS[crc_length]
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length not in _CRC_POLYS:
        raise ValueError(f"Unsupported CRC length: {crc_length}")

    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_remainder(padded, crc_length)

    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否满足 CRC 校验（remainder 为 0）。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length not in _CRC_POLYS:
        raise ValueError(f"Unsupported CRC length: {crc_length}")
    return _crc_remainder(bits, crc_length) == 0


class _SCLPath:
    """单条 SCL 路径。"""

    def __init__(self, N, n, llr_ch):
        self.N = N
        self.n = n
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def clone(self):
        new = _SCLPath(self.N, self.n, np.zeros(self.N))
        new.L = self.L.copy()
        new.B = self.B.copy()
        new.pm = self.pm
        new.u_hat = self.u_hat.copy()
        return new


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_metric_penalty(self, llr_val, u_val):
        """路径度量惩罚：判决与 LLR 符号不一致时加 |LLR|。"""
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_val == hard else abs(llr_val)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_SCLPath(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            candidates = []

            for path in paths:
                _update_llrs_nonrecursive(l, path.L, path.B, self.n, self.N)
                llr_bit = path.L[l, self.n]

                if self.frozen_bits[l]:
                    new_path = path.clone()
                    penalty = self._path_metric_penalty(llr_bit, 0)
                    new_path.pm += penalty
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    _update_bits_nonrecursive(l, new_path.B, self.n, self.N)
                    candidates.append(new_path)
                else:
                    for u_val in (0, 1):
                        new_path = path.clone()
                        penalty = self._path_metric_penalty(llr_bit, u_val)
                        new_path.pm += penalty
                        new_path.u_hat[l] = u_val
                        new_path.B[l, self.n] = u_val
                        _update_bits_nonrecursive(l, new_path.B, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        if self.crc_length > 0:
            valid_paths = []
            for p in paths:
                info = p.u_hat[~self.frozen_bits]
                if crc_check(info, self.crc_length):
                    valid_paths.append(p)
            if valid_paths:
                paths = valid_paths

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
