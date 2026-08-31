"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    _bit_reversed_scalar,
    _active_llr_level,
    _active_bit_level,
    upper_llr_exact,
    lower_llr_exact,
    _map_channel_llrs,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """MSB-first CRC remainder（标准多项式，不含隐式 leading 1 在寄存器外）。"""
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in bits:
        reg ^= int(bit)
        for _ in range(8 if crc_length == 8 else crc_length):
            if reg & 1:
                reg = (reg >> 1) ^ poly
            else:
                reg >>= 1
    return reg & mask


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length)], dtype=int
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
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


# ==================== SCL 译码器 ====================


def _update_llrs_path(L, B, l, n):
    """对单条路径更新 LLR（与 SC 相同）。"""
    N = L.shape[0]
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = upper_llr_exact(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = lower_llr_exact(
                    L[j, s], L[j - branch_size, s], top_bit
                )


def _update_bits_path(B, l, n, bit_val, N):
    """比特回传。"""
    B[l, n] = bit_val
    if l < N / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def _path_metric_penalty(llr_val, u_bit):
    """路径度量惩罚：与 LLR 硬判决不一致时加 |LLR|。"""
    hard = 0 if llr_val >= 0 else 1
    return 0.0 if u_bit == hard else abs(llr_val)


class SCLDecoder:
    """
    SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [
            _bit_reversed_scalar(i, self.n) for i in range(N)
        ]

    def _init_path(self, llr_nat):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan)
        L[:, 0] = llr_nat
        return {"L": L, "B": B, "pm": 0.0}

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat（长度 N），pm（最优路径度量）
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_nat = _map_channel_llrs(llr_ch, self.N)
        paths = [self._init_path(llr_nat)]

        for l in self.decode_order:
            candidates = []
            for path in paths:
                _update_llrs_path(path["L"], path["B"], l, self.n)
                llr_dec = path["L"][l, self.n]

                if l in self.frozen_set:
                    new_path = {
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "pm": path["pm"] + _path_metric_penalty(llr_dec, 0),
                    }
                    _update_bits_path(new_path["B"], l, self.n, 0, self.N)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "pm": path["pm"] + _path_metric_penalty(llr_dec, u_bit),
                        }
                        _update_bits_path(new_path["B"], l, self.n, u_bit, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[:self.list_size]

        # 选择最优路径（CRC 辅助）
        u_candidates = [p["B"][:, self.n].astype(int) for p in paths]
        if self.crc_length > 0:
            info_idx = np.where(self.frozen_bits == 0)[0]
            K_info = len(info_idx) - self.crc_length
            valid = []
            for i, u_hat in enumerate(u_candidates):
                info_bits = u_hat[info_idx][:K_info]
                crc_part = u_hat[info_idx][K_info:]
                check_bits = np.concatenate([info_bits, crc_part])
                if crc_check(check_bits, self.crc_length):
                    valid.append(i)
            if valid:
                best_i = min(valid, key=lambda i: paths[i]["pm"])
            else:
                best_i = 0
        else:
            best_i = 0

        return u_candidates[best_i], paths[best_i]["pm"]
