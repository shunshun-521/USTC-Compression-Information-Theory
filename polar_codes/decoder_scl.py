"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversed
from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level, _hard_decision


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY

    reg = 0
    for b in info_bits:
        reg ^= (int(b) << (crc_length - 1))
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _update_llrs_path(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits_path(B, l, n):
    if l < B.shape[0] // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = (B[j, s] + B[j - branch_size, s]) % 2
                B[j, s - 1] = B[j, s]


# ==================== SCL 译码器 ====================

class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length

    def _pm_update(self, pm, llr, u):
        hard = _hard_decision(llr)
        if u != hard:
            return pm + abs(llr)
        return pm

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        N, n, L = self.N, self.n, self.L
        frozen = self.frozen_bits

        L_arr = np.zeros((N, n + 1), dtype=np.float64)
        B_arr = np.zeros((N, n + 1), dtype=int)
        L_arr[:, 0] = llr_ch

        paths = [{"L": L_arr, "B": B_arr, "pm": 0.0}]

        for i in range(N):
            l = bit_reversed(i, n)
            new_paths = []

            for path in paths:
                P, B, pm = path["L"], path["B"], path["pm"]
                _update_llrs_path(P, B, l, n)
                llr = P[l, n]

                if frozen[l]:
                    u = 0
                    new_pm = self._pm_update(pm, llr, u)
                    B[l, n] = u
                    _update_bits_path(B, l, n)
                    new_paths.append({"L": P, "B": B, "pm": new_pm})
                else:
                    for u in (0, 1):
                        Pc = P.copy()
                        Bc = B.copy()
                        new_pm = self._pm_update(pm, llr, u)
                        Bc[l, n] = u
                        _update_bits_path(Bc, l, n)
                        new_paths.append({"L": Pc, "B": Bc, "pm": new_pm})

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[:L]

        crc_paths = [p for p in paths if self._crc_pass(p["B"][:, n])]
        if crc_paths:
            best = min(crc_paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["B"][:, n], best["pm"]

    def _crc_pass(self, u_hat):
        if self.crc_length == 0:
            return True
        info_idx = np.where(~self.frozen_bits)[0]
        info_bits = u_hat[info_idx]
        return crc_check(info_bits, self.crc_length)
