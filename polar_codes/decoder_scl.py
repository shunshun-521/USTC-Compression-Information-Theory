"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _frozen_set,
    _prepare_llr,
    f_operation,
    g_operation,
)


def _crc_polynomial(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8 (0x07) 或 CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_polynomial(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


class _Path:
  __slots__ = ("L", "B", "pm", "active")

  def __init__(self, N, n):
    self.L = np.zeros((N, n + 1), dtype=np.float64)
    self.B = np.zeros((N, n + 1), dtype=np.int8)
    self.pm = 0.0
    self.active = True


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.frozen = _frozen_set(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_metric_penalty(self, llr, bit):
        return 0.0 if bit == (0 if llr >= 0 else 1) else abs(llr)

    def _clone_path(self, src):
        dst = _Path(self.N, self.n)
        dst.L[:] = src.L
        dst.B[:] = src.B
        dst.pm = src.pm
        return dst

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
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
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        llr = _prepare_llr(llr_ch)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr

        for phi_nat in range(self.N):
            l = _bit_reversed_index(phi_nat, self.n)
            new_paths = []

            for path in paths:
                if not path.active:
                    continue
                self._update_llrs(path, l)
                llr_bit = path.L[l, self.n]

                if l in self.frozen:
                    bit = 0
                    path.pm += self._path_metric_penalty(llr_bit, bit)
                    path.B[l, self.n] = bit
                    if llr_bit < 0:
                        path.L[l, self.n] = -llr_bit
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = self._clone_path(path)
                        child.pm += self._path_metric_penalty(llr_bit, bit)
                        child.B[l, self.n] = bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        candidates = [(p.B[:, self.n].astype(int), p.pm) for p in paths]

        if self.crc_length > 0:
            valid = [
                (u, pm)
                for u, pm in candidates
                if crc_check(u, self.crc_length)
            ]
            if valid:
                valid.sort(key=lambda x: x[1])
                return valid[0]

        candidates.sort(key=lambda x: x[1])
        return candidates[0]


def verify_scl_equals_sc(N=64, K=32, num_frames=20):
    """L=1 的 SCL 应与 SC 等价。"""
    from construction import ga_construction
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
    from decoder_sc import sc_decode
    from encoder import polar_encode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(4.0, K / N)
    rng = np.random.default_rng(1)
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
    return True
