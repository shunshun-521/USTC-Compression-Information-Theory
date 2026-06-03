"""
AWGN 信道 + BPSK 调制/解调（与 SCD 因子图一致的 polarcodes 约定）
"""
import numpy as np

_ES = 1.0


def bpsk_modulate(x):
    """0 -> -sqrt(Es), 1 -> +sqrt(Es)"""
    x = np.asarray(x, dtype=np.float64)
    return 2.0 * (x - 0.5) * np.sqrt(_ES)


def awgn_channel(s, sigma, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return s + rng.normal(0.0, sigma, size=np.shape(s))


def compute_llr(y, sigma):
    """LLR = -2*y*sqrt(Es)/sigma^2（polarcodes 约定）"""
    return -2.0 * np.asarray(y, dtype=np.float64) * np.sqrt(_ES) / (sigma ** 2)


def eb_n0_to_sigma(eb_n0_db, rate):
    snr_linear = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))
    return 1.0 / np.sqrt(snr_linear)


def prepare_decoder_llr(llr_ch, N):
    return np.asarray(llr_ch, dtype=np.float64)


def map_decoder_bits_to_natural(u_hat_dec, N):
    return np.asarray(u_hat_dec, dtype=int)


def prepare_frozen_bits_decoder(frozen_nat, N):
    return np.asarray(frozen_nat, dtype=bool)
