"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation, sc_decode_recursive


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected)


def _pm_penalty(llr_val, u_bit):
    hard = 0 if llr_val >= 0 else 1
    return 0.0 if u_bit == hard else abs(llr_val)


def _x_from_uhat(uhat):
    """由硬判决 uhat 递归构造 x 部分和数组（与 SC 译码一致）。"""
    N = len(uhat)
    if N == 1:
        return np.array([0.0 if uhat[0] == 0 else 1.0])

    half = N // 2
    x_left = _x_from_uhat(uhat[:half])
    x_right = _x_from_uhat(uhat[half:])
    x1 = f_operation(x_left, x_right)
    x = np.zeros(N, dtype=np.float64)
    x[0::2] = x1
    x[1::2] = x_right
    return x


class SCLDecoder:
    """SCL 译码器（多路径递归树搜索）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        if self.list_size == 1:
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = self._scl_paths(llr_ch, self.frozen_bits, 0, [(0.0, np.zeros(self.N, dtype=int))])
        paths.sort(key=lambda p: p[0])
        paths = paths[: self.list_size]

        best_crc = None
        best_any = min(paths, key=lambda p: p[0])

        if self.crc_length > 0:
            for pm, u_hat in paths:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or pm < best_crc[0]:
                        best_crc = (pm, u_hat)
            if best_crc is not None:
                return best_crc[1].copy(), best_crc[0]

        return best_any[1].copy(), best_any[0]

    def _scl_paths(self, y, frozen, offset, paths):
        """多路径递归 SCL（偶/奇分解）。"""
        N = len(y)
        if N == 1:
            candidates = []
            idx = offset
            for pm, u_hat in paths:
                llr_val = y[0]
                if frozen[0]:
                    new_u = u_hat.copy()
                    new_u[idx] = 0
                    candidates.append((pm + _pm_penalty(llr_val, 0), new_u))
                else:
                    for bit in (0, 1):
                        new_u = u_hat.copy()
                        new_u[idx] = bit
                        candidates.append(
                            (pm + _pm_penalty(llr_val, bit), new_u)
                        )
            candidates.sort(key=lambda p: p[0])
            return candidates[: self.list_size]

        u1est = f_operation(y[0::2], y[1::2])
        left_paths = self._scl_paths(u1est, frozen[: N // 2], offset, paths)

        all_results = []
        for pm, u_hat in left_paths:
            uhat1 = u_hat[offset: offset + N // 2]
            u1hp = _x_from_uhat(uhat1)
            u2est = g_operation(f_operation(u1hp, y[0::2]), y[1::2], uhat1)
            sub = self._scl_paths(u2est, frozen[N // 2 :], offset + N // 2, [(pm, u_hat)])
            all_results.extend(sub)

        all_results.sort(key=lambda p: p[0])
        return all_results[: self.list_size]
