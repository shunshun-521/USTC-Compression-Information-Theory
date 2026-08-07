"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import active_bit_level, active_llr_level, f_operation
from encoder import bit_reversed, bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_process(data_bits, poly, width, flush=False):
    state = 0
    mask = (1 << width) - 1
    for bit in data_bits:
        feedback = ((state >> (width - 1)) ^ int(bit)) & 1
        state = (state << 1) & mask
        if feedback:
            state ^= poly
    if flush:
        for _ in range(width):
            feedback = (state >> (width - 1)) & 1
            state = (state << 1) & mask
            if feedback:
                state ^= poly
    return state


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_process(info_bits, poly, crc_length, flush=True)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 中的 CRC 部分是否与信息比特匹配。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    payload = bits[:-crc_length]
    crc_part = bits[-crc_length:]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(crc_part, expected)


def _update_llrs_path(L, B, l, n, N):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                if top_bit == 0:
                    L[j, s + 1] = L[j - branch_size, s] + L[j, s]
                else:
                    L[j, s + 1] = L[j, s] - L[j - branch_size, s]


def _update_bits_path(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.rev = bit_reversal_permutation(N)

        self.info_positions = np.where(~self.frozen_bits)[0]
        if crc_length > 0:
            self.crc_positions = self.info_positions[-crc_length:]
            self.payload_positions = self.info_positions[:-crc_length]
        else:
            self.crc_positions = np.array([], dtype=int)
            self.payload_positions = self.info_positions

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.rev]
        N, n = self.N, self.n

        paths = [{
            'pm': 0.0,
            'L': np.zeros((N, n + 1), dtype=np.float64),
            'B': np.zeros((N, n + 1), dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed(i, n)
            candidates = []

            for path in paths:
                _update_llrs_path(path['L'], path['B'], l, n, N)
                llr = path['L'][l, n]

                if l in self.frozen_set:
                    new_path = {
                        'pm': path['pm'] + self._path_metric_penalty(llr, 0),
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                    }
                    new_path['B'][l, n] = 0
                    _update_bits_path(new_path['B'], l, n, N)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = {
                            'pm': path['pm'] + self._path_metric_penalty(llr, bit),
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                        }
                        new_path['B'][l, n] = bit
                        _update_bits_path(new_path['B'], l, n, N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            crc_paths = []
            for path in paths:
                u_hat = path['B'][:, n]
                payload = u_hat[self.payload_positions]
                crc_part = u_hat[self.crc_positions]
                if np.array_equal(crc_part, crc_encode(payload, self.crc_length)[-self.crc_length:]):
                    crc_paths.append(path)
            if crc_paths:
                best = min(crc_paths, key=lambda p: p['pm'])
                return best['B'][:, n].copy(), best['pm']

        best = min(paths, key=lambda p: p['pm'])
        return best['B'][:, n].copy(), best['pm']


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
    from construction import ga_construction
    from decoder_sc import sc_decode

    rng = np.random.default_rng(1)
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False

    sigma = eb_n0_to_sigma(10.0, K / N)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    print(f"SCL L=1 vs SC mismatches: {mismatches}/50")
