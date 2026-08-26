"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    precompute_sc_indices,
    _llr_to_decision,
    _active_llr_level,
    _active_bit_level,
)
from encoder import _bit_rev_indices


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = CRC8_POLY
    elif crc_length == 16:
        poly = CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器（Permuted SC + Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        _, self.llr_layer_vec, self.bit_layer_vec, self.decode_order = precompute_sc_indices(N)
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _new_path(self, llr_ch):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=int)
        L[:, 0] = llr_ch
        return {"pm": 0.0, "L": L, "B": B, "u_hat": np.zeros(self.N, dtype=int)}

    def _copy_path(self, path):
        return {
            "pm": path["pm"],
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "u_hat": path["u_hat"].copy(),
        }

    def _update_llrs(self, path, step, l):
        for s in self.llr_layer_vec[step]:
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path["L"][j, s + 1] = f_operation(
                        path["L"][j, s], path["L"][j + branch_size, s]
                    )
                else:
                    path["L"][j, s + 1] = g_operation(
                        path["L"][j - branch_size, s],
                        path["L"][j, s],
                        path["B"][j - branch_size, s + 1],
                    )

    def _update_bits(self, path, step, l):
        for s in self.bit_layer_vec[step]:
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path["B"][j - branch_size, s - 1] = (
                        path["B"][j, s] ^ path["B"][j - branch_size, s]
                    )
                    path["B"][j, s - 1] = path["B"][j, s]

    def _penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for step, l in enumerate(self.decode_order):
            candidates = []
            for path in paths:
                self._update_llrs(path, step, l)
                llr_val = path["L"][l, self.n]

                if self.frozen_bits[l]:
                    new_path = self._copy_path(path)
                    new_path["pm"] += self._penalty(llr_val, 0)
                    new_path["u_hat"][l] = 0
                    new_path["B"][l, self.n] = 0
                    self._update_bits(new_path, step, l)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path["pm"] += self._penalty(llr_val, u_bit)
                        new_path["u_hat"][l] = u_bit
                        new_path["B"][l, self.n] = u_bit
                        self._update_bits(new_path, step, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            crc_ok = []
            for path in paths:
                info_bits = path["u_hat"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_ok.append(path)
            best = min(crc_ok if crc_ok else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"], best["pm"]
