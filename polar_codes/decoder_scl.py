"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversed_index
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat", "parent_id", "branch_bit")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.u_hat = np.zeros(N, dtype=int)
        self.parent_id = 0
        self.branch_bit = 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversed_index(i, self.n) for i in range(N)]

    def _ensure_path(self, paths, pid):
        if paths[pid] is None:
            paths[pid] = _Path(self.N, self.n)
        return paths[pid]

    def _copy_path_state(self, paths, src_id, dst_id):
        src = paths[src_id]
        dst = self._ensure_path(paths, dst_id)
        dst.pm = src.pm
        dst.L = src.L.copy()
        dst.B = src.B.copy()
        dst.u_hat = src.u_hat.copy()
        dst.parent_id = src_id
        return dst

    def _update_llrs(self, path, l):
        start_layer = self.n - _active_llr_level(l, self.n)
        for s in range(start_layer, self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        paths = [None] * (self.list_size * 2 + 1)
        root = self._ensure_path(paths, 0)
        root.L[:, 0] = llr_ch
        active = [0]

        for l in self.decode_order:
            candidates = []

            for pid in active:
                path = paths[pid]
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    bit = 0
                    new_pm = path.pm + self._pm_penalty(llr, bit)
                    path.B[l, self.n] = bit
                    path.u_hat[l] = bit
                    self._update_bits(path, l)
                    candidates.append((new_pm, pid, bit, False))
                else:
                    for bit in (0, 1):
                        if bit == path.u_hat[l] and len(candidates) == 0:
                            # fast path not used; always branch
                            pass
                        child_id = len(paths)
                        paths.append(None)
                        child = self._copy_path_state(paths, pid, child_id)
                        child.pm = path.pm + self._pm_penalty(llr, bit)
                        child.B[l, self.n] = bit
                        child.u_hat[l] = bit
                        self._update_bits(child, l)
                        candidates.append((child.pm, child_id, bit, True))

            candidates.sort(key=lambda x: x[0])
            active = [c[1] for c in candidates[: self.list_size]]

        best_pm = float("inf")
        best_path = active[0]
        crc_pass = []

        for pid in active:
            path = paths[pid]
            if self.crc_length > 0:
                info_idx = np.where(self.frozen_bits == 0)[0]
                payload = path.u_hat[info_idx]
                if crc_check(payload, self.crc_length):
                    crc_pass.append((path.pm, pid))
            if path.pm < best_pm:
                best_pm = path.pm
                best_path = pid

        if self.crc_length > 0 and crc_pass:
            crc_pass.sort(key=lambda x: x[0])
            best_path = crc_pass[0][1]
            best_pm = crc_pass[0][0]

        return paths[best_path].u_hat.copy(), best_pm
