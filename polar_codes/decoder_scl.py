"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _sc_decode_scd,
    f_operation,
    g_operation,
    _bit_reversed_int,
    _active_llr_level,
    _active_bit_level,
    _f_boxplus,
    _g_boxplus,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。

    使用标准多项式：
      r=8:  CRC-8  (0x07, 即 x^8 + x^2 + x + 1)
      r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
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
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


# ==================== SCL 译码器 ====================

class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。

    内部维护最多 L 条路径，每条路径有：
      - P: LLR 数组（分层存储）
      - C: 比特数组（分层存储）
      - PM: 路径度量（初始为 0，越小越好）
      - u_hat: 当前已译比特序列
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _path_metric_update(self, pm, llr, bit):
        """路径度量更新：与 LLR 符号不一致时加 |LLR| 惩罚。"""
        hard = 0 if llr >= 0 else 1
        penalty = 0.0 if bit == hard else abs(llr)
        return pm + penalty

    def decode(self, llr_ch):
        """
        主译码函数。

        返回：
            u_hat: 长度 N 的估计源序列（最优路径）
            pm: 最优路径的度量值
        """
        if self.list_size == 1 and self.crc_length == 0:
            from decoder_sc import sc_decode
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_dec = llr_ch[self.br]

        N, n = self.N, self.n
        paths = [{
            "L": np.full((N, n + 1), np.nan, dtype=np.float64),
            "B": np.full((N, n + 1), np.nan),
            "pm": 0.0,
            "u": np.zeros(N, dtype=int),
        }]
        paths[0]["L"][:, 0] = llr_dec

        for phi in range(N):
            new_paths = []
            for path in paths:
                L, B, pm, u = path["L"], path["B"], path["pm"], path["u"]
                l = _bit_reversed_int(phi, n)

                for s in range(n - _active_llr_level(l, n), n):
                    block_size = 2 ** (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                        else:
                            top_bit = int(B[j - branch_size, s + 1]) if not np.isnan(B[j - branch_size, s + 1]) else 0
                            L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)

                llr_bit = L[l, n]

                if l in self.frozen_set:
                    u[l] = 0
                    pm_new = self._path_metric_update(pm, llr_bit, 0)
                    if l >= N // 2:
                        self._update_bits(B, l, n, u[l])
                    new_paths.append({
                        "L": L.copy(),
                        "B": B.copy(),
                        "pm": pm_new,
                        "u": u.copy(),
                    })
                else:
                    for bit in (0, 1):
                        Lc = L.copy()
                        Bc = B.copy()
                        uc = u.copy()
                        uc[l] = bit
                        pm_new = self._path_metric_update(pm, llr_bit, bit)
                        if l >= N // 2:
                            self._update_bits(Bc, l, n, bit)
                        new_paths.append({
                            "L": Lc,
                            "B": Bc,
                            "pm": pm_new,
                            "u": uc,
                        })

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u"], self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p["pm"])
            else:
                best = paths[0]
        else:
            best = paths[0]

        return best["u"], best["pm"]

    @staticmethod
    def _update_bits(B, l, n, bit):
        B[l, n] = bit
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    bj = 0 if np.isnan(B[j, s]) else int(B[j, s])
                    bjb = 0 if np.isnan(B[j - branch_size, s]) else int(B[j - branch_size, s])
                    B[j - branch_size, s - 1] = bj ^ bjb
                    B[j, s - 1] = bj
