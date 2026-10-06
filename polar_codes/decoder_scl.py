"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation, sc_decode_recursive, _partial_sums


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """CRC 校验位附加。"""
    poly = _crc_poly(crc_length)
    reg = 0
    bits = np.asarray(info_bits, dtype=np.int8).ravel()
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验尾部 CRC。"""
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected[-crc_length:], bits[-crc_length:])


class SCLDecoder:
    """基于递归 SC 树的 SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = int(list_size)
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            u = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u, 0.0

        # 每条路径: (pm, u_hat)
        paths = [(0.0, np.zeros(self.N, dtype=np.int8))]

        def recurse(paths_in, llr_node, offset):
            n = len(llr_node)
            if n == 1:
                idx = offset
                out = []
                for pm, u_hat in paths_in:
                    llr0 = llr_node[0]
                    if self.frozen_bits[idx]:
                        hard = 0 if llr0 >= 0 else 1
                        pm2 = pm + (0.0 if hard == 0 else abs(llr0))
                        uh = u_hat.copy()
                        uh[idx] = 0
                        out.append((pm2, uh))
                    else:
                        for bit in (0, 1):
                            hard = 0 if llr0 >= 0 else 1
                            pm2 = pm + (0.0 if bit == hard else abs(llr0))
                            uh = u_hat.copy()
                            uh[idx] = bit
                            out.append((pm2, uh))
                out.sort(key=lambda x: x[0])
                return out[: self.list_size]

            half = n // 2
            llr_left = f_operation(llr_node[:half], llr_node[half:])
            left_paths = recurse(paths_in, llr_left, offset)
            merged = []
            for pm, u_hat in left_paths:
                u_left = u_hat[offset : offset + half]
                s_left = _partial_sums(u_left)
                llr_right = g_operation(llr_node[:half], llr_node[half:], s_left)
                right_paths = recurse([(pm, u_hat)], llr_right, offset + half)
                merged.extend(right_paths)
            merged.sort(key=lambda x: x[0])
            return merged[: self.list_size]

        paths = recurse(paths, llr_ch, 0)

        best_crc = None
        best = paths[0]
        for pm, u_hat in paths:
            if pm < best[0]:
                best = (pm, u_hat)
            if self.crc_length > 0 and crc_check(u_hat[self.info_idx], self.crc_length):
                if best_crc is None or pm < best_crc[0]:
                    best_crc = (pm, u_hat)

        if best_crc is not None:
            best = best_crc
        return best[1].astype(int), float(best[0])


def scl_equivalent_sc(llr, frozen_bits):
    u_scl, _ = SCLDecoder(len(llr), frozen_bits, list_size=1).decode(llr)
    u_sc = sc_decode_recursive(llr, frozen_bits)
    return np.array_equal(u_scl, u_sc)
