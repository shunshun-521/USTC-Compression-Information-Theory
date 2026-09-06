"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _ssc_decode_core,
    f_operation,
    g_operation,
)
from encoder import bit_reversed_index


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_update(reg, bit, width, poly):
    mask = (1 << width) - 1
    topbit = 1 << (width - 1)
    feedback = ((reg ^ (int(bit) << (width - 1))) & topbit) != 0
    return ((reg << 1) ^ (poly if feedback else 0)) & mask


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg = _crc_update(reg, bit, crc_length, poly)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    for bit in bits:
        reg = _crc_update(reg, bit, crc_length, poly)
    return reg == 0


def _path_metric_update(pm, llr, u):
    expected = 0 if llr >= 0 else 1
    return pm if u == expected else pm + abs(llr)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversed_index(i, self.n) for i in range(N)]

    def _init_path(self, llr_ch):
        N, n = self.N, self.n
        L = np.full((N, n + 1), np.nan, dtype=np.float64)
        B = np.full((N, n + 1), np.nan, dtype=np.float64)
        L[:, 0] = llr_ch
        return {"L": L, "B": B, "pm": 0.0, "u_hat": np.zeros(N, dtype=np.int8)}

    def _update_llrs(self, path, l):
        L, B = path["L"], path["B"]
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(L[j, s], L[j - branch_size, s], top_bit)

    def _propagate_bits(self, path, l, bit):
        B = path["B"]
        path["u_hat"][l] = bit
        B[l, self.n] = bit
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._init_path(llr_ch)]

        for l in self.decode_order:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path["L"][l, self.n]
                if l in self.frozen_set:
                    new_path = {
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "pm": _path_metric_update(path["pm"], llr, 0),
                        "u_hat": path["u_hat"].copy(),
                    }
                    self._propagate_bits(new_path, l, 0)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "pm": _path_metric_update(path["pm"], llr, bit),
                            "u_hat": path["u_hat"].copy(),
                        }
                        self._propagate_bits(new_path, l, bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            crc_paths = [
                p
                for p in paths
                if crc_check(p["u_hat"][info_idx], self.crc_length)
            ]
            if crc_paths:
                best = min(crc_paths, key=lambda p: p["pm"])
            else:
                best = min(paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]
