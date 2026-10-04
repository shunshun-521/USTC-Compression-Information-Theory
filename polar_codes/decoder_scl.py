"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import sc_decode, _bit_llr_bp_schedule


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_len):
    mask = (1 << crc_len) - 1
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_len - 1)
        for _ in range(1):
            if reg & (1 << (crc_len - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("pm", "u_hat")

    def __init__(self, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)

    def copy(self):
        p = _Path(len(self.u_hat))
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（路径复制开销小，LLR 由 BP 调度计算）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_penalty(llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N)]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                llr_bit = _bit_llr_bp_schedule(
                    llr_ch, self.frozen_bits, path.u_hat, phi, alpha=1.0
                )
                if self.frozen_bits[phi]:
                    child = path.copy()
                    child.pm += self._pm_penalty(llr_bit, 0)
                    child.u_hat[phi] = 0
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        child = path.copy()
                        child.pm += self._pm_penalty(llr_bit, u)
                        child.u_hat[phi] = u
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best_crc = None
        best_pm = None
        for p in paths:
            if self.crc_length > 0:
                payload = p.u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    if best_crc is None or p.pm < best_crc.pm:
                        best_crc = p
            if best_pm is None or p.pm < best_pm:
                best_pm = p

        chosen = best_crc if best_crc is not None else best_pm
        return chosen.u_hat.copy(), chosen.pm
