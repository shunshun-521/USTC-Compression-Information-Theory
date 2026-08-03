"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation, _xor_combine, _prepare_llr


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in info_bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    for _ in range(crc_length):
        reg = reg << 1
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _compute_bit_llr(llr, n, phi, u_prefix, frozen_bits=None):
    """计算第 phi 位的 LLR（已知前 phi 个比特）"""
    result = [0.0]
    u_prefix = np.asarray(u_prefix, dtype=int)

    def decode_node(y, depth, node):
        if depth == n - 1:
            if node == phi:
                result[0] = y[0]
            if node < phi:
                return np.array([u_prefix[node]], dtype=int)
            if frozen_bits is not None and frozen_bits[node]:
                return np.array([0], dtype=int)
            return np.array([1 if y[0] < 0 else 0], dtype=int)

        half = len(y) // 2
        ly, ry = y[:half], y[half:]
        arr1 = decode_node(f_operation(ly, ry), depth + 1, 2 * node)
        arr2 = decode_node(g_operation(ly, ry, arr1), depth + 1, 2 * node + 1)
        return _xor_combine(arr1, arr2)

    decode_node(llr, 0, 0)
    return result[0]


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N)) + 1
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, u_bit):
        hard = 1 if llr < 0 else 0
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr = _prepare_llr(llr_ch, self.N)
        active_paths = [{'pm': 0.0, 'u': np.zeros(self.N, dtype=int)}]

        for phi in range(self.N):
            new_paths = []
            for path in active_paths:
                llr_phi = _compute_bit_llr(
                    llr, self.n, phi, path['u'][:phi], self.frozen_bits
                )

                if self.frozen_bits[phi]:
                    new_u = path['u'].copy()
                    new_u[phi] = 0
                    new_pm = path['pm'] + self._pm_penalty(llr_phi, 0)
                    new_paths.append({'pm': new_pm, 'u': new_u})
                else:
                    for bit in (0, 1):
                        new_u = path['u'].copy()
                        new_u[phi] = bit
                        new_pm = path['pm'] + self._pm_penalty(llr_phi, bit)
                        new_paths.append({'pm': new_pm, 'u': new_u})

            new_paths.sort(key=lambda p: p['pm'])
            active_paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            valid = [p for p in active_paths
                     if crc_check(p['u'][self.info_indices], self.crc_length)]
            best = min(valid if valid else active_paths, key=lambda p: p['pm'])
        else:
            best = min(active_paths, key=lambda p: p['pm'])

        return best['u'], best['pm']


def verify_scl_equals_sc(N=64, K=32, seed=42):
    """验证 L=1 的 SCL 等价于 SC"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(seed)
    sigma = eb_n0_to_sigma(5.0, K / N)
    scl = SCLDecoder(N, frozen_bits, list_size=1)

    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL(L=1) 与 SC 不等价"


if __name__ == "__main__":
    verify_scl_equals_sc()
    print("SCL 路径度量校验通过")
