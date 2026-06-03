"""
AWGN 信道 + BPSK 调制/解调（与 polarcodes.AWGN 一致）
调制：0 -> -sqrt(Es), 1 -> +sqrt(Es)
LLR = -2*y*sqrt(Es)/N0, N0=1
"""
import numpy as np

_N0 = 1.0


def eb_n0_to_es(eb_n0_db, rate):
    """归一化符号能量 Es = (K/M)*Eb/N0_linear，M=N 时 Es = R*Eb/N0"""
    return rate * (10.0 ** (eb_n0_db / 10.0))


def eb_n0_to_sigma(eb_n0_db, rate):
    """兼容接口：返回每维噪声标准差 sqrt(N0/2)"""
    return np.sqrt(_N0 / 2.0)


def bpsk_modulate(x, es=1.0):
    x = np.asarray(x, dtype=np.float64)
    return 2.0 * (x - 0.5) * np.sqrt(es)


def awgn_channel(s, sigma=None, rng=None, es=None):
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, np.sqrt(_N0 / 2.0), size=np.shape(s))
    return s + noise


def compute_llr(y, sigma=None, es=1.0):
    return -2.0 * np.asarray(y, dtype=np.float64) * np.sqrt(es) / _N0


def prepare_decoder_llr(llr_ch, N):
    return np.asarray(llr_ch, dtype=np.float64)


def map_decoder_bits_to_natural(u_hat_dec, N):
    return np.asarray(u_hat_dec, dtype=int)


def prepare_frozen_bits_decoder(frozen_nat, N):
    return np.asarray(frozen_nat, dtype=bool)
