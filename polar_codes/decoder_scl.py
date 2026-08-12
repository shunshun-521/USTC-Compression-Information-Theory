"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    f_operation,
    g_operation,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_divide(data_bits, poly, crc_len):
    """CRC 除法，返回余数比特"""
    reg = [0] * crc_len
    poly_bits = [(poly >> i) & 1 for i in range(crc_len, -1, -1)]
    for bit in data_bits:
        msb = reg[0]
        reg = reg[1:] + [bit ^ msb]
        if msb:
            reg = [reg[i] ^ poly_bits[i + 1] for i in range(crc_len)]
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    r=8: CRC-8 (0x07); r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_divide(info_bits.tolist(), poly, crc_length)
    return np.concatenate([info_bits, np.array(remainder, dtype=np.int8)])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    if crc_length <= 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _path_metric_update(pm, llr, u):
    """路径度量更新：不一致时加 |LLR| 惩罚"""
    u_hard = 0 if llr >= 0 else 1
    if u != u_hard:
        pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = (
            np.asarray(info_indices, dtype=int)
            if info_indices is not None
            else np.where(~self.frozen_bits)[0]
        )
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _init_paths(self, llr_ch):
        return [{
            "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=np.int8),
            "pm": 0.0,
            "u_hat": np.zeros(self.N, dtype=np.int8),
        }]

    def _update_llrs(self, path, l):
        L, B = path["L"], path["B"]
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        B = path["B"]
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        from encoder import bit_reversal_permutation

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[br]

        paths = self._init_paths(llr_ch)
        paths[0]["L"][:, 0] = llr_ch

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr_val = path["L"][l, self.n]

                if self.frozen_bits[l]:
                    pm = _path_metric_update(path["pm"], llr_val, 0)
                    child = {
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "pm": pm,
                        "u_hat": path["u_hat"].copy(),
                    }
                    child["B"][l, self.n] = 0
                    child["u_hat"][l] = 0
                    self._update_bits(child, l)
                    new_paths.append(child)
                else:
                    for u_cand in (0, 1):
                        pm = _path_metric_update(path["pm"], llr_val, u_cand)
                        child = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "pm": pm,
                            "u_hat": path["u_hat"].copy(),
                        }
                        child["B"][l, self.n] = u_cand
                        child["u_hat"][l] = u_cand
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"], best["pm"]
