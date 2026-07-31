"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _active_bit_level,
    _active_llr_level,
    _bit_reverse_index,
    _prepare_llr,
)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ 0x07) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ 0x8005) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


class SCLDecoder:
    """SCL 译码器（Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def _new_path(self, llr):
        path = {
            'pm': 0.0,
            'u_hat': np.zeros(self.N, dtype=int),
            'L': np.zeros((self.N, self.n + 1), dtype=np.float64),
            'B': np.zeros((self.N, self.n + 1), dtype=int),
            'L_copies': {},
            'B_copies': {},
        }
        path['L'][:, 0] = llr
        return path

    def _copy_path(self, path):
        return {
            'pm': path['pm'],
            'u_hat': path['u_hat'].copy(),
            'L': path['L'],
            'B': path['B'],
            'L_copies': path['L_copies'].copy(),
            'B_copies': path['B_copies'].copy(),
        }

    def _get_L(self, path, key):
        if key not in path['L_copies']:
            i, s = key
            path['L_copies'][key] = path['L'][i, s]
        return path['L_copies'][key]

    def _set_L(self, path, key, value):
        i, s = key
        if key not in path['L_copies']:
            path['L_copies'][key] = path['L'][i, s]
        path['L_copies'][key] = value
        path['L'][i, s] = value

    def _get_B(self, path, key):
        if key not in path['B_copies']:
            i, s = key
            path['B_copies'][key] = path['B'][i, s]
        return path['B_copies'][key]

    def _set_B(self, path, key, value):
        i, s = key
        if key not in path['B_copies']:
            path['B_copies'][key] = path['B'][i, s]
        path['B_copies'][key] = value
        path['B'][i, s] = value

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    val = f_operation(self._get_L(path, (j, s)), self._get_L(path, (j + branch_size, s)))
                    self._set_L(path, (j, s + 1), val)
                else:
                    val = g_operation(
                        self._get_L(path, (j - branch_size, s)),
                        self._get_L(path, (j, s)),
                        self._get_B(path, (j - branch_size, s + 1)),
                    )
                    self._set_L(path, (j, s + 1), val)

    def _propagate_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    left = int(self._get_B(path, (j, s))) ^ int(self._get_B(path, (j - branch_size, s)))
                    self._set_B(path, (j - branch_size, s - 1), left)
                    self._set_B(path, (j, s - 1), self._get_B(path, (j, s)))

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr = _prepare_llr(llr_ch)
        paths = [self._new_path(llr)]

        for i in range(self.N):
            l = _bit_reverse_index(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_bit = self._get_L(path, (l, self.n))

                if l in self.frozen_set:
                    new_path = self._copy_path(path)
                    new_path['pm'] += self._pm_penalty(llr_bit, 0)
                    new_path['u_hat'][l] = 0
                    self._set_B(new_path, (l, self.n), 0)
                    self._propagate_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = self._copy_path(path)
                        new_path['pm'] += self._pm_penalty(llr_bit, u)
                        new_path['u_hat'][l] = u
                        self._set_B(new_path, (l, self.n), u)
                        self._propagate_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[:self.list_size]

        best_crc = None
        best = paths[0]
        if self.crc_length > 0:
            for path in paths:
                info_bits = path['u_hat'][self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or path['pm'] < best_crc['pm']:
                        best_crc = path
        chosen = best_crc if best_crc is not None else best
        return chosen['u_hat'], chosen['pm']


def validate_scl_equals_sc(N=64, K=32, eb_n0_db=5.0, num_frames=20):
    """单路径 SCL 应等价于 SC。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(eb_n0_db, K / N)
    rng = np.random.default_rng(1)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"

    return True


if __name__ == "__main__":
    validate_scl_equals_sc()
    print("SCL validation passed.")
