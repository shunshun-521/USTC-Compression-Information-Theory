"""
AWGN 信道 + BPSK 调制/解调
调制：0 -> +1, 1 -> -1
"""
import numpy as np


def bpsk_modulate(x):
    """将二进制码字 x (0/1) 映射为 BPSK 符号 (+1/-1)"""
    x = np.asarray(x)
    return 1.0 - 2.0 * x.astype(np.float64)


def awgn_channel(s, sigma, rng=None):
    """加高斯白噪声，返回 y = s + n"""
    if rng is None:
        noise = np.random.randn(*np.shape(s))
    else:
        noise = rng.standard_normal(np.shape(s))
    return s + sigma * noise


def compute_llr(y, sigma):
    """
    BPSK-AWGN 信道 LLR: ln P(y|0)/P(y|1) = 2y/sigma^2
    """
    y = np.asarray(y, dtype=np.float64)
    return 2.0 * y / (sigma ** 2)


def eb_n0_to_sigma(eb_n0_db, rate):
    """
    Eb/N0 (dB) -> AWGN 噪声标准差 sigma
    SNR = 2R * 10^{Eb/N0/10}, sigma = 1/sqrt(SNR)
    """
    snr_linear = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))
    return 1.0 / np.sqrt(snr_linear)
