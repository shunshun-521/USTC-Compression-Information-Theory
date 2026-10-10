"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import sc_decode


def _crc_lfsr(info_bits, crc_length, poly):
    """按位 LFSR 计算 CRC 寄存器值。"""
    reg = 0
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & msb:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    for _ in range(crc_length):
        if reg & msb:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = 0x07 if crc_length == 8 else 0x8005
    reg = _crc_lfsr(info_bits, crc_length, poly)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = 0x07 if crc_length == 8 else 0x8005
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


def _codeword_metric(u, llr_ch, frozen_bits):
    u = np.asarray(u, dtype=int).copy()
    u[frozen_bits == 1] = 0
    x = polar_encode(u)
    hard = (llr_ch < 0).astype(int)
    return float(np.sum(np.where(x != hard, np.abs(llr_ch), 0.0)))


def _path_metric_bit(llr_bit, bit):
    pred = 0 if llr_bit >= 0 else 1
    return 0.0 if bit == pred else abs(llr_bit)


class SCLDecoder:
    """
    SCL 译码器：L=1 时等价于 SC；L>1 时在 SC 结果基础上进行列表搜索
    （单比特翻转 + 路径度量），并支持 CRC 筛选。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = max(1, int(list_size))
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        u_sc = sc_decode(llr_ch, self.frozen_bits)
        pm_sc = _codeword_metric(u_sc, llr_ch, self.frozen_bits)

        if self.list_size == 1:
            return u_sc, pm_sc

        candidates = {tuple(u_sc): pm_sc}
        for idx in self.info_indices:
            u_flip = u_sc.copy()
            u_flip[idx] ^= 1
            u_flip[self.frozen_bits == 1] = 0
            key = tuple(u_flip)
            pm = _codeword_metric(u_flip, llr_ch, self.frozen_bits)
            if key not in candidates or pm < candidates[key]:
                candidates[key] = pm

        ranked = sorted(candidates.items(), key=lambda kv: kv[1])[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for key, pm in ranked:
                u = np.array(key, dtype=int)
                payload = u[self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append((u, pm))
            if valid:
                return valid[0][0], valid[0][1]

        best_u = np.array(ranked[0][0], dtype=int)
        return best_u, ranked[0][1]
