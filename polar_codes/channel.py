"""
AWGN 信道 + BPSK 调制/解调
与极化码编码器约定一致：0 -> -sqrt(Es), 1 -> +sqrt(Es)
LLR = -2*y*sqrt(Es)/N0, N0=1
"""
import numpy as np


def _es_from_rate(eb_n0_db, rate):
    return rate * (10.0 ** (eb_n0_db / 10.0))


def bpsk_modulate(x, eb_n0_db=None, rate=0.5):
    """BPSK 调制。"""
    x = np.asarray(x, dtype=np.float64)
    if eb_n0_db is None:
        return 1.0 - 2.0 * x
    es = _es_from_rate(eb_n0_db, rate)
    return 2.0 * (x - 0.5) * np.sqrt(es)


def awgn_channel(s, sigma, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=np.shape(s))
    return s + noise


def compute_llr(y, sigma, eb_n0_db=None, rate=0.5):
    """计算信道 LLR（正号倾向比特 0）。"""
    y = np.asarray(y, dtype=np.float64)
    if eb_n0_db is not None:
        es = _es_from_rate(eb_n0_db, rate)
        return -2.0 * y * np.sqrt(es)
    return 2.0 * y / (sigma ** 2)


def eb_n0_to_sigma(eb_n0_db, rate):
    snr_linear = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))
    return 1.0 / np.sqrt(snr_linear)
