"""
极化码 SC（串行抵消）译码器
树形递归输出码字 s，源比特 u = polar_encode(s)（因 G 自逆）
"""
import math
import numpy as np

from encoder import polar_encode
from polar_ops import f_operation, g_operation


def _sc_decode_codeword(llr, frozen_bits):
    """返回估计码字 s（长度 N）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def hard(llr0):
        return 0 if llr0 >= 0 else 1

    def decode_node(lam, frozen_leaf):
        size = len(lam)
        if size == 1:
            bit = 0 if frozen_leaf[0] else hard(lam[0])
            return np.array([bit], dtype=int)

        half = size // 2
        lam_left = f_operation(lam[:half], lam[half:])
        u_left = decode_node(lam_left, frozen_leaf[:half])
        lam_right = g_operation(lam[:half], lam[half:], u_left)
        u_right = decode_node(lam_right, frozen_leaf[half:])

        s = np.zeros(size, dtype=int)
        s[:half] = u_left ^ u_right
        s[half:] = u_right
        return s

    return decode_node(llr, frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC：返回源比特 u"""
    s = _sc_decode_codeword(llr, frozen_bits)
    return polar_encode(s)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 主入口"""
    return sc_decode_recursive(llr_ch, frozen_bits)
