"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, sc_decode


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """计算 CRC 余数（MSB first）。"""
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


# ==================== SCL 辅助 ====================


def _compute_u_up(u_segment):
    """计算分段 u 的递归重编码 u_up 向量（用于 g 运算）。"""

    def enc_up(u):
        n = len(u)
        if n == 1:
            return u, u
        n2 = n // 2
        u1, u1up = enc_up(u[:n2])
        u2, u2up = enc_up(u[n2:])
        u1x = (u1up.astype(int) ^ u2up.astype(int)).astype(int)
        return np.concatenate([u1, u2]), np.concatenate([u1x, u2up.astype(int)])

    return enc_up(np.asarray(u_segment, dtype=int))[1]


def _llr_at_phi(llr_ch, u_hat, phi):
    """给定已译前缀 u_hat[0:phi]，计算比特 phi 的 LLR。"""
    N = len(llr_ch)
    llr = np.asarray(llr_ch, dtype=np.float64)[bit_reversal_permutation(N)]

    def walk(llr_node, offset):
        n = len(llr_node)
        if n == 1:
            return llr_node[0]
        n2 = n // 2
        if phi < offset + n2:
            return walk(f_operation(llr_node[:n2], llr_node[n2:]), offset)
        u_left = u_hat[offset : offset + n2]
        g_in = g_operation(llr_node[:n2], llr_node[n2:], _compute_u_up(u_left))
        return walk(g_in, offset + n2)

    return walk(llr, 0)


# ==================== SCL 译码器 ====================


class _Path:
    """单条译码路径。"""

    __slots__ = ("pm", "u_hat")

    def __init__(self, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        new_path = _Path(len(self.u_hat))
        new_path.pm = self.pm
        new_path.u_hat = self.u_hat.copy()
        return new_path


class SCLDecoder:
    """
    SCL 译码器（逐比特 LLR + 路径裁剪）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _hard_bit(llr):
        return 0 if llr >= 0 else 1

    @staticmethod
    def _pm_penalty(llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N)]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr = _llr_at_phi(llr_ch, path.u_hat, phi)

                if self.frozen_bits[phi]:
                    new_path = path.copy()
                    new_path.pm += self._pm_penalty(llr, 0)
                    new_path.u_hat[phi] = 0
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = path.copy()
                        new_path.pm += self._pm_penalty(llr, u)
                        new_path.u_hat[phi] = u
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            best = min(valid, key=lambda p: p.pm) if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy(), best.pm


def verify_scl_equals_sc(N=64, K=32, seed=1):
    """L=1 的 SCL 应与 SC 等价。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(5.0, K / N)
    rng = np.random.default_rng(seed)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"

    return True


if __name__ == "__main__":
    verify_scl_equals_sc()
    print("SCL verification passed.")
