"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import sc_decode_recursive, _llr_at_phi

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    top = 1 << crc_length
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & (top - 1)
        else:
            reg = (reg << 1) & (top - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _path_metric_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    return pm + (0.0 if u == hard else abs(llr))


class _Path:
    __slots__ = ("pm", "u_hat")

    def __init__(self, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（基于逐位 LLR 与路径复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode_recursive(llr_ch, self.frozen_bits), 0.0

        paths = [_Path(self.N)]
        for phi in range(self.N):
            new_paths = []
            for path in paths:
                if self.frozen_bits[phi]:
                    llr_phi = _llr_at_phi(llr_ch, self.frozen_bits, path.u_hat, phi)
                    p = _Path(self.N)
                    p.u_hat[:] = path.u_hat
                    p.u_hat[phi] = 0
                    p.pm = _path_metric_update(path.pm, llr_phi, 0)
                    new_paths.append(p)
                else:
                    llr_phi = _llr_at_phi(llr_ch, self.frozen_bits, path.u_hat, phi)
                    for u in (0, 1):
                        p = _Path(self.N)
                        p.u_hat[:] = path.u_hat
                        p.u_hat[phi] = u
                        p.pm = _path_metric_update(path.pm, llr_phi, u)
                        new_paths.append(p)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
