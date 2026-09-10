"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _g_node,
    _to_frozen_set,
    f_operation,
    prepare_channel_llr,
    sc_decode,
)


def _crc_polynomial(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _crc_polynomial(crc_length)
    reg = 0
    shift = crc_length - 1
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        reg ^= bit << shift
        for _ in range(crc_length):
            if reg & (1 << shift):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> (shift - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(bits, expected)


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr_ch
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = _to_frozen_set(frozen_bits)
        self.list_size = max(1, list_size)
        self.crc_length = crc_length
        self.frozen_mask = np.array([1 if i in self.frozen_set else 0 for i in range(N)], dtype=int)
        self.info_indices = np.array(sorted(set(range(N)) - self.frozen_set), dtype=int)

    def _clone(self, path):
        new_path = _Path(self.N, self.n, path.L[:, 0].copy())
        new_path.pm = path.pm
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = _g_node(path.L[j, s], path.L[j - branch_size, s], path.B[j - branch_size, s + 1])

    def _update_bits(self, path, l, bit):
        path.B[l, self.n] = bit
        path.u_hat[l] = bit
        if l >= self.N // 2:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                        path.B[j, s - 1] = path.B[j, s]

    @staticmethod
    def _penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_mask)
            return u_hat, 0.0

        llr_ch = prepare_channel_llr(llr_ch)
        paths = [_Path(self.N, self.n, llr_ch)]

        for phase_idx in range(self.N):
            l = _bit_reversed_index(phase_idx, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = self._clone(path)
                    new_path.pm += 0.0 if llr >= 0 else abs(llr)
                    self._update_bits(new_path, l, 0)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._clone(path)
                        new_path.pm += self._penalty(llr, bit)
                        self._update_bits(new_path, l, bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        decoded = [(p.pm, p.u_hat.copy()) for p in paths]

        if self.crc_length > 0:
            valid = []
            for pm, u_hat in decoded:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append((pm, u_hat))
            if valid:
                valid.sort(key=lambda x: x[0])
                return valid[0][1], valid[0][0]

        decoded.sort(key=lambda x: x[0])
        return decoded[0][1], decoded[0][0]
