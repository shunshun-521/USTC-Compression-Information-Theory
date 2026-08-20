"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, _bit_reversed, _active_llr_level, _active_bit_level
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class Path:
    """单条 SCL 路径（Lazy Copy：L 共享，B 按需复制）。"""

    __slots__ = ("L", "B", "pm", "u_hat", "_owns_L")

    def __init__(self, N, n, llr):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self._owns_L = True

    def fork(self):
        child = Path.__new__(Path)
        child.L = self.L
        child.B = self.B.copy()
        child.pm = self.pm
        child.u_hat = self.u_hat.copy()
        child._owns_L = False
        return child

    def ensure_own_L(self):
        if not self._owns_L:
            self.L = self.L.copy()
            self._owns_L = True


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        frozen_bits = np.asarray(frozen_bits)
        if frozen_bits.dtype != bool:
            frozen_bits = frozen_bits.astype(bool)
        self.frozen_set = set(np.where(frozen_bits)[0])
        self.info_positions = sorted(set(range(N)) - self.frozen_set)
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)

    def _llr_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _update_llrs(self, paths, l):
        n = self.n
        for path in paths:
            path.ensure_own_L()
            L, B = path.L, path.B
            for s in range(n - _active_llr_level(l, n), n):
                block_size = 2 ** (s + 1)
                branch_size = block_size // 2
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                    else:
                        top_bit = int(B[j - branch_size, s + 1])
                        L[j, s + 1] = g_operation(
                            L[j - branch_size, s], L[j, s], top_bit
                        )

    def _update_bits(self, paths, l):
        if l < self.N / 2:
            return
        n = self.n
        for path in paths:
            B = path.B
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2**s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                            B[j - branch_size, s]
                        )
                        B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.rev].copy()
        paths = [Path(self.N, self.n, llr)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            self._update_llrs(paths, l)

            candidates = []
            for path in paths:
                llr_val = path.L[l, self.n]
                if l in self.frozen_set:
                    child = path.fork()
                    child.pm += self._llr_penalty(llr_val, 0)
                    child.u_hat[l] = 0
                    child.B[l, self.n] = 0
                    candidates.append(child)
                else:
                    for bit in (0, 1):
                        child = path.fork()
                        child.pm += self._llr_penalty(llr_val, bit)
                        child.u_hat[l] = bit
                        child.B[l, self.n] = bit
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]
            self._update_bits(paths, l)

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_positions], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    errors = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(8.0, 0.5), rng),
            eb_n0_to_sigma(8.0, 0.5),
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            errors += 1
    print(f"L=1 SCL == SC: {20 - errors}/20 frames match")
