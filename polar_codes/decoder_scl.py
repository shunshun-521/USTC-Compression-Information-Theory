"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import _get_generator_inverse, _frozen_mask, f_operation, g_operation


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=np.int32)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int32)
    if crc_length == 8:
        poly = 0x07
    else:
        poly = 0x8005
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class _Path:
    __slots__ = ("pm", "u", "active")

    def __init__(self, N):
        self.pm = 0.0
        self.u = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器（路径度量 + 列表裁剪）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = _frozen_mask(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.asarray(info_indices) if info_indices is not None else None
        self.Ginv = _get_generator_inverse(N)
        self.bit_llrs = None

    def _bit_llrs(self, llr_ch):
        """由信道 LLR 经 G^{-T} 得到逐比特软信息（用于路径度量）。"""
        return self.Ginv.T @ np.asarray(llr_ch, dtype=np.float64)

    def _path_metric_update(self, pm, llr_bit, u_bit):
        v = 0 if llr_bit >= 0 else 1
        if u_bit != v:
            pm += abs(llr_bit)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            from decoder_sc import sc_decode

            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0
        bit_llrs = self._bit_llrs(llr_ch)
        paths = [_Path(self.N)]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                if not path.active:
                    continue
                if self.frozen_bits[phi]:
                    pm = self._path_metric_update(path.pm, bit_llrs[phi], 0)
                    u_new = path.u.copy()
                    u_new[phi] = 0
                    candidates.append((pm, u_new))
                else:
                    for bit in (0, 1):
                        pm = self._path_metric_update(path.pm, bit_llrs[phi], bit)
                        u_new = path.u.copy()
                        u_new[phi] = bit
                        candidates.append((pm, u_new))

            candidates.sort(key=lambda x: x[0])
            paths = []
            for pm, u in candidates[: self.list_size]:
                p = _Path(self.N)
                p.pm = pm
                p.u = u
                paths.append(p)

        if self.crc_length > 0:
            valid = []
            for p in paths:
                msg = p.u[self.info_indices] if self.info_indices is not None else p.u
                if crc_check(msg, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u, best.pm
