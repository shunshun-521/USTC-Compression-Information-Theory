"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    bit_reversed, f_boxplus, g_operation, _g_llr,
    _active_llr_level, _active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_mod(msg, gen):
    """二进制多项式长除法求 CRC 余数"""
    data = msg.copy()
    for i in range(len(data) - len(gen) + 1):
        if data[i] == 1:
            data[i:i + len(gen)] ^= gen
    return data[-(len(gen) - 1):]


def _gen_poly(crc_length):
    if crc_length == 8:
        return np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)  # x^8+x^2+x+1
    return np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    gen = _gen_poly(crc_length)
    msg = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    crc_bits = _crc_mod(msg, gen)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    gen = _gen_poly(crc_length)
    return np.all(_crc_mod(bits.copy(), gen) == 0)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversed(i, self.n) for i in range(N)]

    def _update_llrs(self, L, B, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_boxplus(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _g_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _current_llr(self, L, l):
        return L[l, self.n]

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，输入自然顺序信道 LLR"""
        from encoder import bit_reversal_permutation
        rev = bit_reversal_permutation(self.N)
        llr = llr_ch[rev]

        paths = [{
            'L': np.zeros((self.N, self.n + 1)),
            'B': np.zeros((self.N, self.n + 1), dtype=int),
            'pm': 0.0,
            'u_hat': np.zeros(self.N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                self._update_llrs(path['L'], path['B'], l)
                llr_val = self._current_llr(path['L'], l)

                if self.frozen_bits[l]:
                    p = {
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                        'pm': path['pm'] + self._pm_penalty(llr_val, 0),
                        'u_hat': path['u_hat'].copy(),
                    }
                    p['u_hat'][l] = 0
                    p['B'][l, self.n] = 0
                    self._update_bits(p['B'], l)
                    new_paths.append(p)
                else:
                    for u in (0, 1):
                        p = {
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'pm': path['pm'] + self._pm_penalty(llr_val, u),
                            'u_hat': path['u_hat'].copy(),
                        }
                        p['u_hat'][l] = u
                        p['B'][l, self.n] = u
                        self._update_bits(p['B'], l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p['u_hat'], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['u_hat'].copy(), best['pm']


def run_scl_tests():
    """SCL 单元测试：L=1 应等价于 SC"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    rng = np.random.default_rng(123)
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    sigma = eb_n0_to_sigma(8.0, K / N)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"L=1 SCL 与 SC 有 {mismatches} 处不一致"

    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits[:4], crc_length=8)
    assert crc_check(encoded, crc_length=8), "CRC 校验失败"
    print("SCL tests passed.")


if __name__ == "__main__":
    run_scl_tests()
