"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from encoder import polar_encode
from decoder_sc import sc_decode


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """CRC 校验位附加到信息比特后"""
    poly = _crc_poly(crc_length)
    info_bits = np.asarray(info_bits, dtype=np.int8)
    reg = 0
    for bit in info_bits:
        fb = ((reg >> (crc_length - 1)) & 1) ^ bit
        reg = ((reg << 1) & ((1 << crc_length) - 1))
        if fb:
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        fb = ((reg >> (crc_length - 1)) & 1) ^ bit
        reg = ((reg << 1) & ((1 << crc_length) - 1))
        if fb:
            reg ^= poly
    return reg == 0


def _path_metric(u, llr_ch):
    """路径度量：与信道 LLR 不一致的比特惩罚之和（越小越好）"""
    x = polar_encode(u)
    hard = (llr_ch < 0).astype(int)
    mismatch = x != hard
    return float(np.sum(np.abs(llr_ch[mismatch])))


class SCLDecoder:
    """SCL 译码器（基于路径分裂 + 码字域路径度量）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = max(1, list_size)
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, _path_metric(u, llr_ch)

        base = sc_decode(llr_ch, self.frozen_bits)
        candidates = {tuple(base): _path_metric(base, llr_ch)}

        info_order = list(self.info_indices)
        if self.crc_length > 0:
            info_order = info_order[: -(self.crc_length)]

        for idx in info_order:
            new_candidates = {}
            for u_tuple, pm in candidates.items():
                u = np.array(u_tuple, dtype=np.int8)
                for bit in (0, 1):
                    u_new = u.copy()
                    u_new[idx] = bit
                    key = tuple(u_new)
                    pm_new = _path_metric(u_new, llr_ch)
                    if key not in new_candidates or pm_new < new_candidates[key]:
                        new_candidates[key] = pm_new
            sorted_paths = sorted(new_candidates.items(), key=lambda x: x[1])
            candidates = dict(sorted_paths[: self.list_size])

        best_u = np.array(min(candidates.items(), key=lambda x: x[1])[0], dtype=int)

        if self.crc_length > 0:
            payload = best_u[self.info_indices]
            if not crc_check(payload, self.crc_length):
                for u_tuple, pm in sorted(candidates.items(), key=lambda x: x[1]):
                    u_try = np.array(u_tuple, dtype=int)
                    if crc_check(u_try[self.info_indices], self.crc_length):
                        best_u = u_try
                        break

        return best_u.astype(int), float(candidates[tuple(best_u)])
