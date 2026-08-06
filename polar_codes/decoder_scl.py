"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation, g_operation, _decide, sc_decode_recursive


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """计算 CRC 余数（MSB first）。"""
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    msg = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    rem = _crc_remainder(msg, poly, crc_length)

    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


class Path:
    """单条 SCL 路径。"""

    __slots__ = ("pm", "llr", "u_hat")

    def __init__(self, llr):
        self.pm = 0.0
        self.llr = llr.copy()
        self.u_hat = None


class SCLDecoder:
    """SCL 译码器（Arikan 递归结构 + 路径度量）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_penalty(self, llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def _scl_decode_block(self, llr, frozen, list_size):
        """对任意长度（2 的幂）子码块做 SCL。"""
        N = len(llr)
        if N == 1:
            paths = [Path(llr)]
            if frozen[0]:
                paths[0].pm += self._path_penalty(llr[0], 0)
                paths[0].u_hat = np.array([0], dtype=int)
            else:
                new_paths = []
                for u_bit in (0, 1):
                    p = Path(llr)
                    p.pm += self._path_penalty(llr[0], u_bit)
                    p.u_hat = np.array([u_bit], dtype=int)
                    new_paths.append(p)
                paths = sorted(new_paths, key=lambda p: p.pm)[:list_size]
            return paths

        if N == 2:
            paths = []
            for u0 in (0, 1):
                if frozen[0] and u0 != 0:
                    continue
                pm0 = self._path_penalty(f_operation(llr[0], llr[1]), u0)
                for u1 in (0, 1):
                    if frozen[1] and u1 != 0:
                        continue
                    pm1 = self._path_penalty(g_operation(llr[0], llr[1], u0), u1)
                    p = Path(llr)
                    p.pm = pm0 + pm1
                    p.u_hat = np.array([u0, u1], dtype=int)
                    paths.append(p)
            paths.sort(key=lambda p: p.pm)
            return paths[:list_size]

        half = N // 2
        l_prime = np.array(
            [f_operation(llr[2 * i], llr[2 * i + 1]) for i in range(half)], dtype=np.float64
        )
        left_paths = self._scl_decode_block(l_prime, frozen[:half], list_size)

        all_paths = []
        for lp in left_paths:
            v = polar_encode(lp.u_hat)
            l_double = np.array(
                [
                    g_operation(llr[2 * i], llr[2 * i + 1], v[i])
                    for i in range(half)
                ],
                dtype=np.float64,
            )
            right_paths = self._scl_decode_block(l_double, frozen[half:], list_size)
            for rp in right_paths:
                merged = Path(llr)
                merged.pm = lp.pm + rp.pm
                merged.u_hat = np.concatenate([lp.u_hat, rp.u_hat])
                all_paths.append(merged)

        all_paths.sort(key=lambda p: p.pm)
        return all_paths[:list_size]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = self._scl_decode_block(llr_ch, self.frozen_bits, self.list_size)

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p.pm)
            else:
                best = min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm


def scl_decode_equivalent_to_sc(N, frozen_bits, llr_ch):
    """L=1 的 SCL 应与 SC 等价。"""
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    u_scl, _ = scl.decode(llr_ch)
    u_sc = sc_decode_recursive(llr_ch, frozen_bits)
    return np.array_equal(u_scl, u_sc)
