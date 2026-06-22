"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _align_channel_llrs,
    f_operation,
    g_operation,
)
from encoder import bit_reversed_index


_CRC_POLYS = {8: 0x07, 16: 0x8005}


def _crc_encode_raw(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).flatten()
    poly = _CRC_POLYS[crc_length]
    reg = 0
    for bit in info_bits:
        reg ^= bit << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    return _crc_encode_raw(info_bits, crc_length)


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int).flatten()
    return np.array_equal(bits, _crc_encode_raw(bits[:-crc_length], crc_length))


class Path:
    """单条 SCL 译码路径（Lazy Copy）。"""

    __slots__ = ('L', 'B', 'pm', 'active')

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.active = True

    def clone(self):
        new_path = Path.__new__(Path)
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        new_path.pm = self.pm
        new_path.active = True
        return new_path


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversed_index(i, self.n) for i in range(N)]
        self.llr_layers = []
        self.bit_layers = []
        for l in self.decode_order:
            start = self.n - _active_llr_level(l, self.n)
            self.llr_layers.append(list(range(start, self.n)))
            if l < N // 2:
                self.bit_layers.append([])
            else:
                start_bit = self.n - _active_bit_level(l, self.n)
                self.bit_layers.append(list(range(self.n, start_bit, -1)))

    def _update_llrs(self, paths, phi):
        l = self.decode_order[phi]
        layers = self.llr_layers[phi]
        for path in paths:
            if not path.active:
                continue
            for s in layers:
                block_size = 1 << (s + 1)
                branch_size = block_size // 2
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                    else:
                        path.L[j, s + 1] = g_operation(
                            path.L[j - branch_size, s],
                            path.L[j, s],
                            path.B[j - branch_size, s + 1],
                        )

    def _update_bits(self, path, phi, bit):
        l = self.decode_order[phi]
        path.B[l, self.n] = bit
        for s in self.bit_layers[phi]:
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = _align_channel_llrs(llr_ch)
        paths = [Path(self.N, self.n, llr_ch)]
        u_hat_partial = {0: np.zeros(self.N, dtype=int)}

        for phi in range(self.N):
            l = self.decode_order[phi]
            self._update_llrs(paths, phi)

            candidates = []
            for p_idx, path in enumerate(paths):
                if not path.active:
                    continue
                llr = path.L[l, self.n]
                if self.frozen_bits[l]:
                    bit = 0
                    new_path = path.clone()
                    new_path.pm += self._pm_penalty(llr, 0)
                    self._update_bits(new_path, phi, 0)
                    u_hat_partial[id(new_path)] = u_hat_partial.get(id(path), np.zeros(self.N, dtype=int)).copy()
                    u_hat_partial[id(new_path)][l] = 0
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = path.clone()
                        new_path.pm += self._pm_penalty(llr, bit)
                        self._update_bits(new_path, phi, bit)
                        u = u_hat_partial.get(id(path), np.zeros(self.N, dtype=int)).copy()
                        u[l] = bit
                        u_hat_partial[id(new_path)] = u
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]
            u_hat_partial = {id(p): u_hat_partial[id(p)] for p in paths if id(p) in u_hat_partial}

        best = min(paths, key=lambda p: p.pm)
        u_hat = np.zeros(self.N, dtype=int)
        for l in self.decode_order:
            u_hat[l] = best.B[l, self.n]

        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            crc_pass = []
            for path in paths:
                u = np.zeros(self.N, dtype=int)
                for l in self.decode_order:
                    u[l] = path.B[l, self.n]
                payload = u[info_positions]
                if crc_check(payload, self.crc_length):
                    crc_pass.append(path)
            if crc_pass:
                best = min(crc_pass, key=lambda p: p.pm)
                for l in self.decode_order:
                    u_hat[l] = best.B[l, self.n]

        return u_hat, best.pm
