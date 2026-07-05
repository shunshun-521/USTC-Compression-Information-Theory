"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _prepare_llr,
    _to_frozen_mask,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in bits:
        fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = ((reg << 1) | fb) & mask
        if fb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    reg = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    return _crc_remainder(bits, crc_length) == 0


def _hard_bit_from_llr(llr):
    return 0 if llr >= 0 else 1


def _path_metric_penalty(llr, bit):
    expected = _hard_bit_from_llr(llr)
    return 0.0 if bit == expected else abs(llr)


class _Path:
    """单条 SCL 路径，Lazy Copy 通过写时复制 L/B 实现。"""

    __slots__ = ("L", "B", "pm", "_L_owner", "_B_owner")

    def __init__(self, N, n, llr_ch=None):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        if llr_ch is not None:
            self.L[:, 0] = llr_ch
        self.pm = 0.0
        self._L_owner = True
        self._B_owner = True

    def fork(self):
        child = _Path(self.L.shape[0], self.L.shape[1] - 1)
        child.L = self.L
        child.B = self.B
        child.pm = self.pm
        child._L_owner = False
        child._B_owner = False
        return child

    def ensure_L_writable(self):
        if not self._L_owner:
            self.L = self.L.copy()
            self._L_owner = True

    def ensure_B_writable(self):
        if not self._B_owner:
            self.B = self.B.copy()
            self._B_owner = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = _to_frozen_mask(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        path.ensure_L_writable()
        L, B = path.L, path.B
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        path.ensure_B_writable()
        B = path.B
        n = self.n
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = _prepare_llr(llr_ch)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                cur_llr = path.L[l, self.n]
                branch_bits = [0] if self.frozen_bits[l] else [0, 1]

                for bit in branch_bits:
                    new_path = path.fork()
                    new_path.pm += _path_metric_penalty(cur_llr, bit)
                    new_path.ensure_B_writable()
                    new_path.B[l, self.n] = bit
                    self._update_bits(new_path, l)
                    candidates.append((new_path.pm, new_path))

            candidates.sort(key=lambda x: x[0])
            paths = [p for _, p in candidates[: self.list_size]]

        best_path = min(paths, key=lambda p: p.pm)
        u_hat = best_path.B[:, self.n].astype(int)

        if self.crc_length > 0:
            valid = []
            for pm, path in sorted(((p.pm, p) for p in paths), key=lambda x: x[0]):
                cand = path.B[:, self.n].astype(int)
                info = cand[self.info_indices]
                if crc_check(info, self.crc_length):
                    valid.append((pm, cand))
            if valid:
                pm, u_hat = valid[0]
                return u_hat, pm

        return u_hat, best_path.pm


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    u = np.zeros(N, dtype=int)
    u[info_idx] = np.random.randint(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(10.0, 0.5))

    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    print("L=1 matches SC:", np.array_equal(u_sc, u_scl))
