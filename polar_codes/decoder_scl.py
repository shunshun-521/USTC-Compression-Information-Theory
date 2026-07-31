"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from scipy.special import logsumexp

from decoder_sc import f_operation, g_operation, _bit_reversed, _active_llr_level, _active_bit_level


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    top = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


# ==================== SCL 译码器 ====================


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _path_metric_update(self, pm, llr, bit):
        """路径度量更新：与 LLR 符号一致不惩罚，否则加 |LLR|。"""
        hard = 0 if llr >= 0 else 1
        penalty = 0.0 if bit == hard else abs(llr)
        return pm + penalty

    def decode(self, llr_ch):
        """主译码函数，返回最优路径的估计序列与路径度量。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [
            {
                "pm": 0.0,
                "L": np.full((N, n + 1), np.nan, dtype=np.float64),
                "B": np.full((N, n + 1), np.nan),
                "u_hat": np.zeros(N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for phi in [_bit_reversed(i, n) for i in range(N)]:
            new_paths = []
            for path in paths:
                L, B = path["L"], path["B"]
                for s in range(n - _active_llr_level(phi, n), n):
                    block_size = 2 ** (s + 1)
                    branch_size = block_size // 2
                    for j in range(phi, N, block_size):
                        if j % block_size < branch_size:
                            L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                        else:
                            L[j, s + 1] = g_operation(
                                L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                            )

                llr_bit = L[phi, n]
                if phi in self.frozen_set:
                    bit = 0
                    pm = self._path_metric_update(path["pm"], llr_bit, bit)
                    child = {
                        "pm": pm,
                        "L": L.copy(),
                        "B": B.copy(),
                        "u_hat": path["u_hat"].copy(),
                    }
                    child["B"][phi, n] = bit
                    child["u_hat"][phi] = bit
                    self._update_bits(child, phi)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        pm = self._path_metric_update(path["pm"], llr_bit, bit)
                        child = {
                            "pm": pm,
                            "L": L.copy(),
                            "B": B.copy(),
                            "u_hat": path["u_hat"].copy(),
                        }
                        child["B"][phi, n] = bit
                        child["u_hat"][phi] = bit
                        self._update_bits(child, phi)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_passes(p["u_hat"])]
            if valid:
                best = min(valid, key=lambda p: p["pm"])
            else:
                best = min(paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"], best["pm"]

    def _update_bits(self, path, phi):
        B = path["B"]
        n = self.n
        N = self.N
        if phi >= N / 2:
            for s in range(n, n - _active_bit_level(phi, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(phi, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                            B[j - branch_size, s]
                        )
                        B[j, s - 1] = B[j, s]

    def _crc_passes(self, u_hat):
        info_bits = u_hat[self.info_indices]
        if self.crc_length <= 0:
            return True
        k_info = len(info_bits) - self.crc_length
        if k_info <= 0:
            return False
        payload = info_bits[:k_info]
        attached = info_bits[k_info:]
        expected = crc_encode(payload, self.crc_length)[-self.crc_length :]
        return np.array_equal(attached, expected)
