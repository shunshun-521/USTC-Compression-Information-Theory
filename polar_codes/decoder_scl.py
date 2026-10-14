"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _SCDState,
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    poly = _crc_poly(crc_length)
    info_bits = np.asarray(info_bits, dtype=np.int8)
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


class _SCLPath:
    __slots__ = ("pm", "state")

    def __init__(self, state, pm=0.0):
        self.pm = pm
        self.state = state


class SCLDecoder:
    """SCL 译码器（基于 Vangala 置换 SC）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        init_state = _SCDState(self.N, self.n, llr_ch, self.frozen_bits)
        paths = [_SCLPath(init_state, 0.0)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                path.state.update_llrs(l)
                llr0 = path.state.L[l, self.n]

                if self.frozen_bits[l]:
                    pen = self._pm_penalty(llr0, 0)
                    path.pm += pen
                    path.state.B[l, self.n] = 0
                    path.state.update_bits(l)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        st = _SCDState(self.N, self.n, llr_ch, self.frozen_bits)
                        st.L = path.state.L.copy()
                        st.B = path.state.B.copy()
                        st.B[l, self.n] = u_bit
                        st.update_bits(l)
                        new_paths.append(
                            _SCLPath(st, path.pm + self._pm_penalty(llr0, u_bit))
                        )

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        candidates = paths
        if self.crc_length > 0:
            passed = []
            for p in candidates:
                info_bits = p.state.B[:, self.n][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    passed.append(p)
            if passed:
                candidates = passed

        best = min(candidates, key=lambda p: p.pm)
        return best.state.B[:, self.n].astype(int), best.pm
