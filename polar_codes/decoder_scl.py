"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversed_index
from decoder_sc import (
    active_bit_level,
    active_llr_level,
    lower_llr,
    upper_llr,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8).flatten()
    if crc_length == 8:
        poly, init = _CRC8_POLY, 0
    elif crc_length == 16:
        poly, init = _CRC16_POLY, 0
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = init
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8).flatten()
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


# ==================== SCL 译码器 ====================


class SCLDecoder:
    """SCL 译码器（Lazy Copy 路径管理）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        """
        返回：
            u_hat: 最优路径估计（长度 N）
            pm: 最优路径度量
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, L = self.N, self.n, self.list_size

        paths = [{
            "L": np.zeros((N, n + 1), dtype=np.float64),
            "B": np.zeros((N, n + 1), dtype=np.float64),
            "u": np.zeros(N, dtype=np.int8),
            "pm": 0.0,
            "active": True,
        }]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed_index(i, n)
            new_paths = []

            for path in paths:
                if not path["active"]:
                    continue
                self._update_llrs(path, l)

                llr_dec = path["L"][l, n]

                if self.frozen_bits[l]:
                    candidates = [(0, self._path_metric(path["pm"], llr_dec, 0))]
                else:
                    candidates = [
                        (0, self._path_metric(path["pm"], llr_dec, 0)),
                        (1, self._path_metric(path["pm"], llr_dec, 1)),
                    ]

                for bit, pm in candidates:
                    child = self._fork_path(path)
                    child["pm"] = pm
                    child["u"][l] = bit
                    child["B"][l, n] = bit
                    if l >= N // 2:
                        self._update_bits(child, l)
                    new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[:L]
            if not paths:
                paths = [self._fork_path(new_paths[0])]

        return self._select_best(paths)

    def _path_metric(self, pm, llr, bit):
        """与 LLR 符号一致不惩罚，否则加 |LLR|。"""
        hard = 0 if llr >= 0 else 1
        if bit == hard:
            return pm
        return pm + abs(llr)

    def _fork_path(self, path):
        """复制路径状态（列表较小时拷贝 LLR 数组以保证正确性）。"""
        return {
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "u": path["u"].copy(),
            "pm": path["pm"],
            "active": True,
        }

    def _update_llrs(self, path, l):
        L, B = path["L"], path["B"]
        n = self.n
        N = self.N
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    if np.isnan(L[j, s]):
                        continue
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s], use_min_sum=False)
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], top_bit, use_min_sum=False
                    )

    def _update_bits(self, path, l):
        B = path["B"]
        n = self.n
        N = self.N
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _select_best(self, paths):
        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            valid = []
            for p in paths:
                payload = p["u"][info_idx]
                if crc_check(payload, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid
        best = min(paths, key=lambda p: p["pm"])
        return best["u"].astype(int), best["pm"]
