"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed,
    _frozen_to_set,
    _update_llrs,
    _update_bits,
)
from encoder import bit_reversed_index


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_bits(poly, order, data_bits):
    """按 MSB 优先计算 CRC 校验位。"""
    reg = 0
    for b in data_bits:
        reg ^= int(b) << (order - 1)
        if reg & (1 << (order - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << order) - 1)
        else:
            reg = (reg << 1) & ((1 << order) - 1)
    return [(reg >> (order - 1 - i)) & 1 for i in range(order)]


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    r=8: CRC-8 (0x07); r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        crc = _crc_bits(_CRC8_POLY, 8, info_bits)
    elif crc_length == 16:
        crc = _crc_bits(_CRC16_POLY, 16, info_bits)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, np.array(crc, dtype=int)])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected)


# ==================== SCL 译码器 ====================


class SCLDecoder:
    """SCL 译码器（路径复制 + 路径度量）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = _frozen_to_set(frozen_bits, N)
        self.info_set = sorted(set(range(N)) - self.frozen_set)

    def _path_penalty(self, llr, u):
        """与 LLR 符号不一致时加 |LLR| 惩罚。"""
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        """
        SCL 译码。

        返回：
            u_hat: 最优路径估计（长度 N）
            pm: 最优路径度量
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [
            {
                "pm": 0.0,
                "L": np.zeros((N, n + 1), dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=np.int8),
                "u": np.zeros(N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []

            for path in paths:
                L, B, pm = path["L"], path["B"], path["pm"]
                _update_llrs(L, B, l, n, N)
                cur_llr = L[l, n]

                if l in self.frozen_set:
                    u0 = 0
                    pm_new = pm + self._path_penalty(cur_llr, u0)
                    B[l, n] = u0
                    path["u"][l] = u0
                    _update_bits(B, l, n, N)
                    new_paths.append(
                        {"pm": pm_new, "L": L.copy(), "B": B.copy(), "u": path["u"].copy()}
                    )
                else:
                    for u_cand in (0, 1):
                        Lc = L.copy()
                        Bc = B.copy()
                        uc = path["u"].copy()
                        pm_new = pm + self._path_penalty(cur_llr, u_cand)
                        Bc[l, n] = u_cand
                        uc[l] = u_cand
                        _update_bits(Bc, l, n, N)
                        new_paths.append(
                            {"pm": pm_new, "L": Lc, "B": Bc, "u": uc}
                        )

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u"][self.info_set], self.crc_length)]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u"], best["pm"]
