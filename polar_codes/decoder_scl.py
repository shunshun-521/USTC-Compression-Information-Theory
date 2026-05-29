"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _prepare_llr,
    _update_bit_layers,
    _update_llr_layers,
    precompute_sc_indices,
    sc_decode,
)


def _crc_polynomial(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_polynomial(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(
        bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    )


class Path:
    """SCL 路径（Lazy Copy）"""

    __slots__ = ("P", "C", "pm", "u_hat", "_owned")

    def __init__(self, n, N):
        self.P = np.zeros((n + 1, N), dtype=np.float64)
        self.C = np.zeros((n + 1, N), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self._owned = True

    def fork(self):
        new = Path.__new__(Path)
        new.P = self.P
        new.C = self.C
        new.pm = self.pm
        new.u_hat = self.u_hat.copy()
        new._owned = False
        return new

    def ensure_owned(self):
        if not self._owned:
            self.P = self.P.copy()
            self.C = self.C.copy()
            self._owned = True


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)

    def _pm_penalty(self, llr, u_bit):
        llr = np.clip(llr, -30.0, 30.0)
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr = _prepare_llr(llr_ch)
        paths = [Path(self.n, self.N)]
        paths[0].P[self.n, :] = llr

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                path.ensure_owned()
                _update_llr_layers(path.P, path.C, self.llr_layer_vec[phi], self.N)
                llr_root = path.P[0, 0]

                if self.frozen_bits[phi]:
                    cand = path.fork()
                    cand.ensure_owned()
                    cand.pm += self._pm_penalty(llr_root, 0)
                    cand.u_hat[phi] = 0
                    cand.C[0, 0] = 0
                    _update_bit_layers(cand.C, self.bit_layer_vec[phi], self.N)
                    new_paths.append(cand)
                else:
                    for u_bit in (0, 1):
                        cand = path.fork()
                        cand.ensure_owned()
                        cand.pm += self._pm_penalty(llr_root, u_bit)
                        cand.u_hat[phi] = u_bit
                        cand.C[0, 0] = u_bit
                        _update_bit_layers(cand.C, self.bit_layer_vec[phi], self.N)
                        new_paths.append(cand)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
