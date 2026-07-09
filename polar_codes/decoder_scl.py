"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _frozen_set_from_array,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, width):
    mask = (1 << width) - 1
    crc = 0
    for bit in bits:
        crc ^= int(bit) << (width - 1)
        for _ in range(1):
            if crc & (1 << (width - 1)):
                crc = ((crc << 1) ^ poly) & mask
            else:
                crc = (crc << 1) & mask
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits.tolist(), poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(bits.tolist(), poly, crc_length)
    return rem == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = _frozen_set_from_array(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.array(sorted(set(range(N)) - self.frozen_set), dtype=int)

    @staticmethod
    def _update_llrs(L, B, l, n, N):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    @staticmethod
    def _update_bits(B, l, n, N):
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    @staticmethod
    def _metric_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.full((N, n + 1), np.nan, dtype=np.float64)
        B = np.full((N, n + 1), np.nan)
        L[:, 0] = llr_ch

        paths = [{"L": L, "B": B, "pm": 0.0}]

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []

            for path in paths:
                self._update_llrs(path["L"], path["B"], l, n, N)
                llr = path["L"][l, n]

                if l in self.frozen_set:
                    child = path if not new_paths else {
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "pm": path["pm"],
                    }
                    child["pm"] = child["pm"] + self._metric_penalty(llr, 0)
                    child["B"][l, n] = 0
                    self._update_bits(child["B"], l, n, N)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "pm": path["pm"] + self._metric_penalty(llr, bit),
                        }
                        child["B"][l, n] = bit
                        self._update_bits(child["B"], l, n, N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        candidates = []
        for path in paths:
            u = np.zeros(N, dtype=int)
            for l in range(N):
                val = path["B"][l, n]
                if not np.isnan(val):
                    u[l] = int(val)
            candidates.append((path["pm"], u))

        if self.crc_length > 0:
            crc_ok = []
            for pm, u in candidates:
                info = u[self.info_indices]
                if crc_check(info, self.crc_length):
                    crc_ok.append((pm, u))
            if crc_ok:
                crc_ok.sort(key=lambda x: x[0])
                return crc_ok[0][1], crc_ok[0][0]

        candidates.sort(key=lambda x: x[0])
        return candidates[0][1], candidates[0][0]
