"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import channel_llr_to_decoder, compute_bit_llr

import crcmod

_CRC8 = crcmod.mkCrcFun(0x107, initCrc=0, rev=False, xorOut=0)
_CRC16 = crcmod.mkCrcFun(0x11021, initCrc=0, rev=False, xorOut=0)


def _bits_to_bytes(bits):
    bits = np.asarray(bits, dtype=int)
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=int)])
    return bytes(np.packbits(bits))


def _remainder_to_bits(value, crc_length):
    return np.array(
        [(value >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    crc_fn = _CRC8 if crc_length == 8 else _CRC16
    remainder = crc_fn(_bits_to_bytes(info_bits))
    crc_bits = _remainder_to_bits(remainder, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    crc_fn = _CRC8 if crc_length == 8 else _CRC16
    return crc_fn(_bits_to_bytes(bits)) == 0


class _Path:
    """单条 SCL 路径"""

    __slots__ = ("pm", "u_hat")

    def __init__(self, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        new_path = _Path(len(self.u_hat))
        new_path.pm = self.pm
        new_path.u_hat = self.u_hat.copy()
        return new_path


class SCLDecoder:
    """SCL 译码器（Lazy Copy：仅复制 u_hat 与路径度量）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _path_metric_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = channel_llr_to_decoder(llr_ch)
        paths = [_Path(self.N)]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr = compute_bit_llr(llr_ch, path.u_hat, phi, self.N)
                if self.frozen_bits[phi]:
                    new_path = path.copy()
                    new_path.pm += self._path_metric_penalty(llr, 0)
                    new_path.u_hat[phi] = 0
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = path.copy()
                        new_path.pm += self._path_metric_penalty(llr, bit)
                        new_path.u_hat[phi] = bit
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_crc = None
        if self.crc_length > 0:
            for path in paths:
                payload = path.u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    if best_crc is None or path.pm < best_crc.pm:
                        best_crc = path

        best = best_crc if best_crc is not None else paths[0]
        return best.u_hat.copy(), best.pm
