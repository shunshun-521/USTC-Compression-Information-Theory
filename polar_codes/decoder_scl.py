"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import _prepare_llr, _as_frozen_mask, llr_at_phase


def _crc_polynomial(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_polynomial(crc_length)
    reg = 0
    for bit in info_bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


class SCLDecoder:
    """SCL 译码器：逐相位路径扩展 + Lazy Copy（复制 u 前缀）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = _as_frozen_mask(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _metric(llr, bit):
        if (bit == 0 and llr >= 0) or (bit == 1 and llr < 0):
            return 0.0
        return abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            from decoder_sc import sc_decode_recursive
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = _prepare_llr(llr_ch)
        paths = [(0.0, np.zeros(0, dtype=int))]

        for phi in range(self.N):
            new_paths = []
            for pm, prefix in paths:
                llr = llr_at_phase(llr_ch, self.frozen_bits, prefix, phi)
                if self.frozen_bits[phi]:
                    bit = 0
                    new_prefix = np.concatenate([prefix, [bit]])
                    new_paths.append((pm + self._metric(llr, bit), new_prefix))
                else:
                    for bit in (0, 1):
                        new_prefix = np.concatenate([prefix, [bit]])
                        new_paths.append((pm + self._metric(llr, bit), new_prefix))

            new_paths.sort(key=lambda x: x[0])
            paths = new_paths[: self.list_size]

        candidates = []
        for pm, prefix in paths:
            u_hat = np.zeros(self.N, dtype=int)
            u_hat[: len(prefix)] = prefix
            candidates.append((pm, u_hat))

        if self.crc_length > 0:
            valid = [
                (pm, u)
                for pm, u in candidates
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            if valid:
                pm, u_hat = min(valid, key=lambda x: x[0])
                return u_hat, pm

        pm, u_hat = min(candidates, key=lambda x: x[0])
        return u_hat, pm
