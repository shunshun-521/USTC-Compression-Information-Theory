"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed_index,
    sc_decode,
)
from encoder import bit_reversal_permutation

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_process(bits, poly, width, flush=False):
    """按位 CRC 处理（MSB 优先）。"""
    state = 0
    mask = (1 << width) - 1
    for bit in bits:
        msb = (state >> (width - 1)) & 1
        state = ((state << 1) | int(bit)) & mask
        if msb:
            state ^= poly
    if flush:
        for _ in range(width):
            msb = (state >> (width - 1)) & 1
            state = (state << 1) & mask
            if msb:
                state ^= poly
    return state


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8 (0x07) 或 CRC-16 (0x8005)。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_process(info_bits, poly, crc_length, flush=True)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_process(bits, poly, crc_length, flush=True) == 0


# ==================== SCL 译码器 ====================


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed_index(i, self.n) for i in range(N)]
        self._rev = bit_reversal_permutation(N)

    def _path_metric_update(self, pm, llr, u):
        """路径度量更新：与 LLR 符号不一致时加 |LLR| 惩罚。"""
        preferred = 0 if llr >= 0 else 1
        penalty = 0.0 if u == preferred else abs(llr)
        return pm + penalty

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, pm)。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self._rev]
        N, n = self.N, self.n

        paths = [{
            'L': np.zeros((N, n + 1), dtype=np.float64),
            'B': np.zeros((N, n + 1), dtype=int),
            'pm': 0.0,
            'u_hat': np.zeros(N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for step, l in enumerate(self.decode_order):
            new_paths = []
            for path in paths:
                L, B = path['L'], path['B']
                for s in range(n - _active_llr_level(l, n), n):
                    block_size = 1 << (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                        else:
                            L[j, s + 1] = g_operation(
                                L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                            )

                llr_leaf = L[l, n]
                if self.frozen_bits[l]:
                    child = {
                        'L': L.copy(),
                        'B': B.copy(),
                        'pm': self._path_metric_update(path['pm'], llr_leaf, 0),
                        'u_hat': path['u_hat'].copy(),
                    }
                    child['u_hat'][l] = 0
                    child['B'][l, n] = 0
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        child = {
                            'L': L.copy(),
                            'B': B.copy(),
                            'pm': self._path_metric_update(path['pm'], llr_leaf, u),
                            'u_hat': path['u_hat'].copy(),
                        }
                        child['u_hat'][l] = u
                        child['B'][l, n] = u
                        new_paths.append(child)

            for path in new_paths:
                l = self.decode_order[step]
                B = path['B']
                if l >= N // 2:
                    for s in range(n, n - _active_bit_level(l, n), -1):
                        block_size = 1 << s
                        branch_size = block_size // 2
                        for j in range(l, -1, -block_size):
                            if j % block_size >= branch_size:
                                B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                                B[j, s - 1] = B[j, s]

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            info_mask = self.frozen_bits == 0
            valid = [p for p in paths if crc_check(p['u_hat'][info_mask], self.crc_length)]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p['pm'])
        return best['u_hat'], best['pm']


def scl_equals_sc(N, frozen_bits, llr_ch):
    """验证 L=1 时 SCL 等价于 SC。"""
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    u_scl, _ = scl.decode(llr_ch)
    u_sc = sc_decode(llr_ch, frozen_bits)
    return np.array_equal(u_scl, u_sc)


if __name__ == "__main__":
    from construction import ga_construction
    from channel import bpsk_modulate

    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        from encoder import polar_encode
        x = polar_encode(u)
        llr = 50.0 * bpsk_modulate(x)
        assert scl_equals_sc(N, frozen, llr), "L=1 SCL 与 SC 不等价"
    print("SCL L=1 等价性测试通过")
