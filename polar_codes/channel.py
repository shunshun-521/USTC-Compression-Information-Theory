"""
AWGN 信道 + BPSK 调制/解调
"""
import numpy as np


def bpsk_modulate(x):
    """0 -> -1, 1 -> +1（与 SCD 一致）"""
    return 2.0 * (np.asarray(x, dtype=np.float64) - 0.5)


def awgn_channel(s, sigma, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return s + rng.normal(0.0, sigma, size=s.shape)


def compute_llr(y, sigma):
    """LLR = -2y/sigma^2，正值倾向 bit 0"""
    return -2.0 * np.asarray(y, dtype=np.float64) / (sigma ** 2)


def eb_n0_to_sigma(eb_n0_db, rate):
    snr_linear = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))
    return 1.0 / np.sqrt(snr_linear)


def align_llr_for_decoder(llr_ch):
    return np.asarray(llr_ch, dtype=np.float64)
