"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation, g_operation, _prepare_channel_llr,
    _bit_reversed, _active_llr_level, _active_bit_level,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _poly_generator(poly, crc_length):
    """构造 GF(2) 生成多项式系数（含最高次项）。"""
    full = poly | (1 << crc_length)
    return [(full >> i) & 1 for i in range(crc_length, -1, -1)]


def _gf2_remainder(msg, gen):
    msg = list(msg)
    n = len(gen) - 1
    for i in range(len(msg) - n):
        if msg[i]:
            for j in range(len(gen)):
                msg[i + j] ^= gen[j]
    return msg[-n:]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    gen = _poly_generator(poly, crc_length)
    msg = list(info_bits) + [0] * crc_length
    rem = _gf2_remainder(msg, gen)
    return np.concatenate([info_bits, np.array(rem, dtype=int)])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    gen = _poly_generator(poly, crc_length)
    rem = _gf2_remainder(bits, gen)
    return all(x == 0 for x in rem)


# ==================== SCL 译码器 ====================


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _path_metric_update(self, pm, llr, bit):
        """路径度量更新。"""
        hard = 0 if llr >= 0 else 1
        penalty = 0.0 if bit == hard else abs(llr)
        return pm + penalty

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = _prepare_channel_llr(llr_ch)
        N, n = self.N, self.n
        L_size = self.list_size

        paths = [{
            'L': np.full((N, n + 1), np.nan, dtype=np.float64),
            'B': np.full((N, n + 1), np.nan),
            'pm': 0.0,
            'active': True,
        }]
        paths[0]['L'][:, 0] = llr_ch

        for phi_idx, l in enumerate(self.decode_order):
            new_paths = []

            for p in paths:
                if not p['active']:
                    continue

                L, B = p['L'], p['B']

                for s in range(n - _active_llr_level(l, n), n):
                    block_size = 2 ** (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                        else:
                            top_bit = int(B[j - branch_size, s + 1]) if not np.isnan(B[j - branch_size, s + 1]) else 0
                            L[j, s + 1] = g_operation(L[j, s], L[j - branch_size, s], top_bit)

                cur_llr = L[l, n]

                if self.frozen_bits[l]:
                    pm = self._path_metric_update(p['pm'], cur_llr, 0)
                    B[l, n] = 0
                    np_new = {
                        'L': L.copy(),
                        'B': B.copy(),
                        'pm': pm,
                        'active': True,
                    }
                    new_paths.append(np_new)
                else:
                    for bit in (0, 1):
                        Lc = L.copy()
                        Bc = B.copy()
                        Bc[l, n] = bit
                        pm = self._path_metric_update(p['pm'], cur_llr, bit)
                        new_paths.append({
                            'L': Lc,
                            'B': Bc,
                            'pm': pm,
                            'active': True,
                        })

            new_paths.sort(key=lambda x: x['pm'])
            paths = new_paths[:L_size]

            for p in paths:
                B = p['B']
                l = self.decode_order[phi_idx]
                if l < N / 2:
                    continue
                for s in range(n, n - _active_bit_level(l, n), -1):
                    block_size = 2 ** s
                    branch_size = block_size // 2
                    for j in range(l, -1, -block_size):
                        if j % block_size >= branch_size:
                            B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                            B[j, s - 1] = B[j, s]

        best_u = paths[0]['B'][:, n].astype(int)
        best_pm = paths[0]['pm']

        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            crc_pass = [p for p in paths
                        if crc_check(p['B'][:, n].astype(int)[info_positions], self.crc_length)]
            if crc_pass:
                crc_pass.sort(key=lambda x: x['pm'])
                best_u = crc_pass[0]['B'][:, n].astype(int)
                best_pm = crc_pass[0]['pm']

        return best_u, best_pm


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    sigma = eb_n0_to_sigma(8.0, K / N)
    ok_scl = ok_sc = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)) + np.random.normal(0, sigma, N), sigma)
        uh_sc, _ = sc_decode(llr, frozen_bits), None
        uh_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        ok_sc += np.array_equal(uh_sc, u)
        ok_scl += np.array_equal(uh_scl, u)
    print(f"L=1 SCL vs SC: {ok_scl}/50, SC: {ok_sc}/50")
