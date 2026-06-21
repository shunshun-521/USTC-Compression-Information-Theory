"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _llr_to_bit,
    _map_channel_llr,
    f_operation,
    g_operation,
    precompute_sc_indices,
    sc_decode,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


# ==================== SCL 译码器 ====================


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order, _ = precompute_sc_indices(N)

    def _path_metric_update(self, pm, llr, u):
        penalty = 0.0 if u == (0 if llr >= 0 else 1) else abs(llr)
        return pm + penalty

    def _update_llrs(self, L, B, l):
        start = self.n - _active_llr_level(l, self.n)
        for s in range(start, self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        end = self.n - _active_bit_level(l, self.n)
        for s in range(self.n, end, -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数"""
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_work = _map_channel_llr(llr_ch, self.N)
        paths = [
            {
                "pm": 0.0,
                "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
                "B": np.zeros((self.N, self.n + 1), dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr_work

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                self._update_llrs(path["L"], path["B"], l)
                llr_l = path["L"][l, self.n]

                if self.frozen_bits[l]:
                    pm = self._path_metric_update(path["pm"], llr_l, 0)
                    new_p = {
                        "pm": pm,
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                    }
                    new_p["B"][l, self.n] = 0
                    self._update_bits(new_p["B"], l)
                    new_paths.append(new_p)
                else:
                    for u in (0, 1):
                        pm = self._path_metric_update(path["pm"], llr_l, u)
                        new_p = {
                            "pm": pm,
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                        }
                        new_p["B"][l, self.n] = u
                        self._update_bits(new_p["B"], l)
                        new_paths.append(new_p)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            crc_pass = []
            for p in paths:
                u_hat = p["B"][:, self.n]
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(p)
            if crc_pass:
                paths = crc_pass

        best = min(paths, key=lambda p: p["pm"])
        return best["B"][:, self.n], best["pm"]
