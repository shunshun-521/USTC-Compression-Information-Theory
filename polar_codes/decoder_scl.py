"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import itertools

import numpy as np

from decoder_sc import sc_decode
from encoder import polar_encode


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _path_metric(llr_ch, x_bits):
    x_bits = np.asarray(x_bits, dtype=int)
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    signs = 1.0 - 2.0 * x_bits
    return float(np.sum(np.where(signs * llr_ch >= 0, 0.0, np.abs(llr_ch))))


class SCLDecoder:
    """
    SCL 译码器：基于不确定比特翻转的列表译码。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = max(1, list_size)
        self.crc_length = crc_length

    def _enumerate_candidates(self, llr_ch):
        x_base = (llr_ch < 0).astype(int)
        base_pm = _path_metric(llr_ch, x_base)
        candidates = [(base_pm, x_base.copy())]

        if self.list_size == 1:
            return candidates

        num_flips = min(int(np.ceil(np.log2(self.list_size))), 6)
        uncertain = np.argsort(np.abs(llr_ch))[:num_flips]

        for r in range(1, num_flips + 1):
            for flip_idx in itertools.combinations(uncertain, r):
                x_cand = x_base.copy()
                x_cand[list(flip_idx)] ^= 1
                pm = _path_metric(llr_ch, x_cand)
                candidates.append((pm, x_cand))
                if len(candidates) >= self.list_size * 4:
                    break
            if len(candidates) >= self.list_size * 4:
                break

        candidates.sort(key=lambda item: item[0])
        return candidates[: self.list_size]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        candidates = self._enumerate_candidates(llr_ch)
        paths = []
        for pm, x_cand in candidates:
            u_hat = polar_encode(x_cand)
            paths.append((pm, u_hat))

        crc_valid = []
        for pm, u_hat in paths:
            if self.crc_length > 0:
                info_bits = u_hat[~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    crc_valid.append((pm, u_hat))
        if crc_valid:
            best = min(crc_valid, key=lambda item: item[0])
        else:
            best = min(paths, key=lambda item: item[0])
        return best[1].copy(), best[0]
