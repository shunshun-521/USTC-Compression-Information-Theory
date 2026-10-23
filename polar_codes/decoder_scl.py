"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），置换 SC 与编码器配套
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _hard_decision,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """CRC 校验位附加到信息比特后（MSB first）"""
    poly = _crc_poly(crc_length)
    reg = 0
    bits = np.asarray(info_bits, dtype=np.int8).ravel()
    for b in bits:
        reg ^= (int(b) << (crc_length - 1))
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    out = np.zeros(len(bits) + crc_length, dtype=np.int8)
    out[: len(bits)] = bits
    for i in range(crc_length):
        out[len(bits) + i] = (reg >> (crc_length - 1 - i)) & 1
    return out


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    poly = _crc_poly(crc_length)
    reg = 0
    for b in np.asarray(bits, dtype=np.int8).ravel():
        reg ^= (int(b) << (crc_length - 1))
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（路径复制实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] + path.B[j - branch_size, s]
                    ) % 2
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_add(self, pm, llr, u):
        u_from_llr = 0 if llr >= 0 else 1
        if u == u_from_llr:
            return pm
        return pm + abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if l in self.frozen_set:
                    pm = self._pm_add(path.pm, llr, 0)
                    new_path = self._clone_path(path)
                    new_path.pm = pm
                    new_path.B[l, self.n] = 0
                    new_path.u_hat[l] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        pm = self._pm_add(path.pm, llr, u)
                        new_path = self._clone_path(path)
                        new_path.pm = pm
                        new_path.B[l, self.n] = u
                        new_path.u_hat[l] = u
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_crc = None
        best_pm = None
        for path in paths:
            if self.crc_length > 0:
                info_bits = path.u_hat[self.info_indices]
                if not crc_check(info_bits, self.crc_length):
                    continue
            if best_pm is None or path.pm < best_pm:
                best_pm = path.pm
                best_crc = path

        if best_crc is None:
            best_crc = min(paths, key=lambda p: p.pm)
            best_pm = best_crc.pm

        return best_crc.u_hat.astype(int), best_pm

    def _clone_path(self, path):
        new = _Path(self.N, self.n)
        new.L[:] = path.L
        new.B[:] = path.B
        new.pm = path.pm
        new.u_hat[:] = path.u_hat
        return new


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr
    from decoder_sc import sc_decode

    N = 64
    frozen = np.zeros(N, bool)
    frozen[: N // 2] = True
    u = np.zeros(N, int)
    u[~frozen] = np.random.randint(0, 2, N // 2)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-6)
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl)
    print("L=1 SCL matches SC")
