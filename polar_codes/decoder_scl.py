"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _SCDState,
    bit_reversed,
    active_llr_level,
    active_bit_level,
    upper_llr,
    lower_llr,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """信息比特后附加 CRC 校验位（MSB-first 逐比特）"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器（基于 Vangala SC 状态，路径复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def _path_metric_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{"state": _SCDState(self.N, self.frozen_bits, llr_ch), "pm": 0.0}]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                st = path["state"]
                st.update_llrs(l)
                llr = st.L[l, self.n]
                if self.frozen_bits[i]:
                    pen = self._path_metric_penalty(llr, 0)
                    st2 = self._copy_state(st)
                    st2.B[l, self.n] = 0
                    st2.update_bits(l)
                    new_paths.append({"state": st2, "pm": path["pm"] + pen})
                else:
                    for u in (0, 1):
                        st2 = self._copy_state(st)
                        st2.B[l, self.n] = u
                        st2.update_bits(l)
                        pen = self._path_metric_penalty(llr, u)
                        new_paths.append({"state": st2, "pm": path["pm"] + pen})

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            valid = []
            for p in paths:
                u_hat = self._state_to_uhat(p["state"])
                info_bits = u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p["pm"])

        return self._state_to_uhat(best["state"]), best["pm"]

    def _copy_state(self, st):
        c = _SCDState(self.N, self.frozen_bits, st.L[:, 0])
        c.L = st.L.copy()
        c.B = st.B.copy()
        return c

    def _state_to_uhat(self, st):
        decoded = st.B[:, self.n].astype(int)
        out = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            out[i] = decoded[bit_reversed(i, self.n)]
        return out
