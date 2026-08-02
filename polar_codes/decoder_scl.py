"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import scl_decode_core


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_step(reg, bit, poly, crc_length):
    reg ^= int(bit) << (crc_length - 1)
    if reg & (1 << (crc_length - 1)):
        reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
    else:
        reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg = _crc_step(reg, bit, poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in bits:
        reg = _crc_step(reg, bit, poly, crc_length)
    return reg == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr = np.asarray(llr_ch, dtype=np.float64)
        paths = scl_decode_core(llr, self.frozen_bits, self.list_size)

        if self.crc_length > 0:
            valid = []
            for pm, u_hat, _ in paths:
                payload = u_hat[self.info_idx]
                if crc_check(payload, self.crc_length):
                    valid.append((pm, u_hat))
            if valid:
                pm, u_hat = min(valid, key=lambda x: x[0])
                return u_hat, pm

        pm, u_hat, _ = paths[0]
        return u_hat, pm


def verify_scl_equals_sc(N=64, K=32, eb_n0_db=8.0):
    """单路径 SCL 应等价于 SC"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(eb_n0_db, K / N)
    rng = np.random.default_rng(1)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
    return True


if __name__ == "__main__":
    verify_scl_equals_sc()
    print("SCL verification passed")
