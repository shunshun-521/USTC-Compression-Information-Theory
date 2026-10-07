"""
AWGN 信道 + BPSK 调制/解调
与极化码文献一致：0 -> -1, 1 -> +1；LLR = -2y/Es
"""
import numpy as np

ENERGY_PER_SYMBOL = 1.0


def bpsk_modulate(x):
    """0 -> -sqrt(Es), 1 -> +sqrt(Es)"""
    x = np.asarray(x, dtype=np.float64)
    return 2.0 * (x - 0.5) * np.sqrt(ENERGY_PER_SYMBOL)


def awgn_channel(s, sigma, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    s = np.asarray(s, dtype=np.float64)
    n = rng.normal(0.0, sigma, size=s.shape)
    return s + n


def compute_llr(y, sigma):
    """LLR = ln P(y|0)/P(y|1) = -2 y sqrt(Es) / sigma^2"""
    y = np.asarray(y, dtype=np.float64)
    noise_var = sigma ** 2
    return -2.0 * y * np.sqrt(ENERGY_PER_SYMBOL) / noise_var


def eb_n0_to_sigma(eb_n0_db, rate):
    snr_linear = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))
    return 1.0 / np.sqrt(snr_linear)
