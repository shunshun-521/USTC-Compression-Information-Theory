"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    bit_reversed,
    active_llr_level,
    active_bit_level,
    f_operation,
    g_operation,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8 (0x07) 或 CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(
        crc_encode(bits[:-crc_length], crc_length)[-crc_length:],
        bits[-crc_length:],
    )


class PathState:
    """单条译码路径状态（Lazy Copy）"""

    __slots__ = ("pm", "L", "B", "u_hat", "parent", "copy_L", "copy_B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.parent = None
        self.copy_L = False
        self.copy_B = False

    def fork(self):
        child = PathState.__new__(PathState)
        child.pm = self.pm
        child.L = self.L
        child.B = self.B
        child.u_hat = self.u_hat.copy()
        child.parent = self
        child.copy_L = False
        child.copy_B = False
        return child

    def ensure_L_writable(self):
        if not self.copy_L:
            self.L = self.L.copy()
            self.copy_L = True

    def ensure_B_writable(self):
        if not self.copy_B:
            self.B = self.B.copy()
            self.copy_B = True


def _update_llrs_path(path, l, n):
    L, B = path.L, path.B
    for s in range(n - active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, len(L), block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], top_bit
                )


def _update_bits_path(path, l, n, N):
    if l < N // 2:
        return
    B = path.B
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                path.ensure_B_writable()
                B = path.B
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def _pm_penalty(llr_val, bit):
    """路径度量惩罚：与 LLR 符号不一致时加 |LLR|"""
    hard = 0 if llr_val >= 0 else 1
    return 0.0 if hard == bit else abs(llr_val)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        frozen = self.frozen_bits

        paths = [PathState(N, n, llr_ch)]

        for phase in range(N):
            l = bit_reversed(phase, n)
            new_paths = []

            for path in paths:
                path.ensure_L_writable()
                _update_llrs_path(path, l, n)
                llr_val = path.L[l, n]

                if frozen[l]:
                    pen = _pm_penalty(llr_val, 0)
                    path.pm += pen
                    path.ensure_B_writable()
                    path.B[l, n] = 0
                    path.u_hat[l] = 0
                    _update_bits_path(path, l, n, N)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = path.fork()
                        child.pm += _pm_penalty(llr_val, bit)
                        child.ensure_B_writable()
                        child.B[l, n] = bit
                        child.u_hat[l] = bit
                        _update_bits_path(child, l, n, N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[~frozen]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.astype(int), best.pm


def verify_scl_equals_sc(N=64, K=32, num_frames=20):
    """L=1 时 SCL 应等价于 SC"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
    return True
