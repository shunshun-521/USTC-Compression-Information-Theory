"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _psc_update_bits,
    _psc_update_llrs,
)
from encoder import bit_reversal_permutation


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc8_bytes(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    data = np.packbits(info_bits, bitorder="big").tobytes()
    val = _crc8_bytes(data) if crc_length == 8 else _crc16_bytes(data)
    crc_bits = np.array(
        [(val >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def _crc16_bytes(data: bytes) -> int:
    crc = 0
    poly = CRC_POLYNOMIALS[16]
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    data = np.packbits(bits, bitorder="big").tobytes()
    if crc_length == 8:
        return _crc8_bytes(data) == 0
    return _crc16_bytes(data) == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, n, N):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.u_hat = None


class SCLDecoder:
    """SCL 译码器（Lazy Copy：分裂时复制 L/B）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        init = _Path(n, N)
        init.L[:, 0] = llr_ch[bit_reversal_permutation(N)]
        init.u_hat = np.zeros(N, dtype=int)
        paths = [init]

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []

            for path in paths:
                _psc_update_llrs(path.L, path.B, l, n)
                llr = path.L[l, n]

                if l in self.frozen_set:
                    path.pm += self._pm_penalty(llr, 0)
                    path.B[l, n] = 0
                    path.u_hat[l] = 0
                    _psc_update_bits(path.B, l, n)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        child = _Path(n, N)
                        child.L[:] = path.L
                        child.B[:] = path.B
                        child.pm = path.pm + self._pm_penalty(llr, u_bit)
                        child.u_hat = path.u_hat.copy()
                        child.B[l, n] = u_bit
                        child.u_hat[l] = u_bit
                        _psc_update_bits(child.B, l, n)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            info_bits = lambda p: p.u_hat[~self.frozen_bits]
            crc_ok = [p for p in paths if crc_check(info_bits(p), self.crc_length)]
            best = min(crc_ok, key=lambda p: p.pm) if crc_ok else min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat, best.pm
