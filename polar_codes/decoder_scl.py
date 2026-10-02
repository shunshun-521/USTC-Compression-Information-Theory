"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
L=1 时使用标准 SC；L>1 时使用基于 SC 的候选翻转 + 相关度选择（启发式列表译码）
"""
import numpy as np

from channel import bpsk_modulate
from decoder_sc import sc_decode
from encoder import polar_encode


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """信息比特后附加 CRC"""
    info_bits = np.asarray(info_bits, dtype=np.int_)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    for _ in range(crc_length):
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int_
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """校验 CRC"""
    bits = np.asarray(bits, dtype=np.int_)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg == 0


class SCLDecoder:
    """SCL / CA-SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        fb = np.asarray(frozen_bits, dtype=np.int_)
        self.frozen_set = set(np.where(fb == 1)[0])
        self.L = max(1, int(list_size))
        self.crc_length = crc_length
        self.info_pos = np.array(
            [i for i in range(N) if i not in self.frozen_set], dtype=np.int64
        )
        self._fb = np.zeros(N, dtype=np.int_)
        for idx in self.frozen_set:
            self._fb[idx] = 1

    def _score(self, u_hat, llr_ch):
        x = polar_encode(u_hat)
        return float(np.dot(llr_ch, bpsk_modulate(x)))

    def _select_best(self, candidates, llr_ch):
        if self.crc_length > 0:
            valid = [
                u
                for u in candidates
                if crc_check(u[self.info_pos], self.crc_length)
            ]
            if valid:
                candidates = valid
        return max(candidates, key=lambda u: self._score(u, llr_ch))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        u_sc = sc_decode(llr_ch, self._fb)
        if self.L == 1:
            return u_sc, 0.0

        candidates = [u_sc.copy()]
        # 按信道 LLR 在信息位上的幅度选取最不可靠的比特尝试翻转
        order = self.info_pos[np.argsort(np.abs(llr_ch[self.info_pos]))]
        for idx in order:
            if len(candidates) >= self.L:
                break
            u_alt = u_sc.copy()
            u_alt[idx] ^= 1
            candidates.append(u_alt)

        # 若 L 较大，补充双比特翻转组合（仅限较小列表）
        if len(candidates) < self.L and len(order) >= 2:
            for i in range(min(len(order), 6)):
                for j in range(i + 1, min(len(order), 6)):
                    if len(candidates) >= self.L:
                        break
                    u_alt = u_sc.copy()
                    u_alt[order[i]] ^= 1
                    u_alt[order[j]] ^= 1
                    candidates.append(u_alt)

        best = self._select_best(candidates, llr_ch)
        return best, 0.0
