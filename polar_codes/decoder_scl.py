"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, sc_decode_recursive

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, degree):
    reg = 0
    for b in bits:
        reg ^= int(b) << (degree - 1)
        for _ in range(degree):
            top = reg & (1 << (degree - 1))
            reg = (reg << 1) & ((1 << degree) - 1)
            if top:
                reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        rem = _crc_remainder(info_bits, _CRC8_POLY, 8)
        crc_bits = np.array([(rem >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        rem = _crc_remainder(info_bits, _CRC16_POLY, 16)
        crc_bits = np.array([(rem >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    if crc_length == 8:
        return _crc_remainder(bits, _CRC8_POLY, 8) == 0
    if crc_length == 16:
        return _crc_remainder(bits, _CRC16_POLY, 16) == 0
    return False


def _pm_leaf(llr, u):
    hard = 0 if llr >= 0 else 1
    return 0.0 if u == hard else abs(llr)


def _scl_rec(llr_blk, frozen_blk, list_size):
    """递归 SCL，返回路径列表 [(pm, u_hat, u_up), ...]。"""
    n = len(llr_blk)
    frozen_blk = np.asarray(frozen_blk, dtype=bool)

    if n == 1:
        paths = []
        if frozen_blk[0]:
            paths.append((0.0, np.array([0], dtype=int), np.array([0.0])))
        else:
            for bit in (0, 1):
                pm = _pm_leaf(llr_blk[0], bit)
                paths.append((pm, np.array([bit], dtype=int), np.array([float(bit)])))
        return paths

    n2 = n // 2
    l1, l2 = llr_blk[:n2], llr_blk[n2:]
    f1, f2 = frozen_blk[:n2], frozen_blk[n2:]

    paths_left = _scl_rec(f_operation(l1, l2), f1, list_size)
    all_paths = []

    for pm_l, u_l, u_l_up in paths_left:
        llr_r = g_operation(l1, l2, u_l_up)
        paths_right = _scl_rec(llr_r, f2, list_size)
        for pm_r, u_r, u_r_up in paths_right:
            pm = pm_l + pm_r
            u_hat = np.concatenate([u_l, u_r])
            u_l_up_new = (u_l_up.astype(int) ^ u_r_up.astype(int)).astype(np.float64)
            u_up = np.concatenate([u_l_up_new, u_r_up])
            all_paths.append((pm, u_hat, u_up))

    all_paths.sort(key=lambda x: x[0])
    return all_paths[:list_size]


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.L == 1 and self.crc_length == 0:
            u = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u, 0.0

        paths = _scl_rec(llr_ch, self.frozen_bits, self.L)

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p[1][self.info_positions], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda x: x[0])
        return best[1], best[0]


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False

    rng = np.random.default_rng(1)
    err_sc = err_scl = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        sigma = eb_n0_to_sigma(8.0, 0.5)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        if not np.array_equal(u[info_idx], sc_decode(llr, frozen)[info_idx]):
            err_sc += 1
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u[info_idx], u_scl[info_idx]):
            err_scl += 1
    print(f"L=1 vs SC: sc_err={err_sc}, scl_err={err_scl}")
