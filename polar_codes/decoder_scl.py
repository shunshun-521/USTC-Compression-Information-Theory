"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    _prepare_llr,
    _active_llr_level,
    _active_bit_level,
    _update_llrs,
    _update_bits,
)
from encoder import bit_reversal_permutation

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_generator_bits(crc_length):
    """CRC 生成多项式比特（含最高位 1）。"""
    if crc_length == 8:
        return [1, 0, 0, 0, 0, 0, 1, 1, 1]
    return [1] + [0] * 14 + [1, 0, 1]


def _crc_mod(bits, gen_bits):
    """多项式模运算求余数。"""
    msg = list(bits)
    k = len(gen_bits) - 1
    for i in range(len(bits)):
        if msg[i] == 1:
            for j in range(len(gen_bits)):
                if i + j < len(msg):
                    msg[i + j] ^= gen_bits[j]
    return msg[-k:]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    gen = _crc_generator_bits(crc_length)
    remainder = _crc_mod(list(info_bits) + [0] * crc_length, gen)
    return np.concatenate([info_bits, np.array(remainder, dtype=int)])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    gen = _crc_generator_bits(crc_length)
    remainder = _crc_mod(bits, gen)
    return all(b == 0 for b in remainder)


# ==================== SCL 译码器 ====================


class _Path:
    __slots__ = ('L', 'B', 'pm', 'u_hat')

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [bit_reversal_permutation(N)[i] for i in range(N)]

    @staticmethod
    def _metric_penalty(llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        llr_ch = _prepare_llr(llr_ch)
        N, n = self.N, self.n
        paths = [_Path(N, n, llr_ch)]

        for l in self.decode_order:
            candidates = []
            for path in paths:
                _update_llrs(path.L, path.B, l, n)
                llr_val = path.L[l, n]

                if self.frozen_bits[l]:
                    new_path = _Path(N, n, llr_ch)
                    new_path.L[:] = path.L
                    new_path.B[:] = path.B
                    new_path.pm = path.pm + self._metric_penalty(llr_val, 0)
                    new_path.u_hat[:] = path.u_hat
                    new_path.u_hat[l] = 0
                    new_path.B[l, n] = 0
                    _update_bits(new_path.B, l, n)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = _Path(N, n, llr_ch)
                        new_path.L[:] = path.L
                        new_path.B[:] = path.B
                        new_path.pm = path.pm + self._metric_penalty(llr_val, u_bit)
                        new_path.u_hat[:] = path.u_hat
                        new_path.u_hat[l] = u_bit
                        new_path.B[l, n] = u_bit
                        _update_bits(new_path.B, l, n)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = min(valid, key=lambda p: p.pm) if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy(), best.pm


if __name__ == '__main__':
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(1)
    u_sent = np.zeros(N, dtype=int)
    u_sent[info_idx] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u_sent)), eb_n0_to_sigma(10.0, K / N))

    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    print('L=1 SCL matches SC:', np.array_equal(u_sc, u_scl))
