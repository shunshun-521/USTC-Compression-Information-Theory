"""
AWGN 信道 + BPSK 调制/解调
调制：0 -> +1, 1 -> -1
LLR = 2*y/sigma^2 = ln P(y|0)/P(y|1)
"""
import numpy as np


def bpsk_modulate(x):
    """将二进制码字 x (0/1) 映射为 BPSK 符号 (+1/-1)"""
    return 1.0 - 2.0 * np.asarray(x, dtype=np.float64)


def awgn_channel(s, sigma, rng=None):
    """加高斯白噪声，返回接收信号 y = s + n，n ~ N(0, sigma^2)"""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=s.shape)
    return s + noise


def compute_llr(y, sigma):
    """BPSK-AWGN 信道 LLR：LLR(y) = 2*y / sigma^2"""
    return 2.0 * y / (sigma**2)


def eb_n0_to_sigma(eb_n0_db, rate):
    """Eb/N0 (dB) -> 噪声标准差 sigma"""
    snr_linear = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))
    return 1.0 / np.sqrt(snr_linear)


def hard_decision_llr(llr):
    """LLR 硬判决：LLR >= 0 -> 0, else 1"""
    return (llr < 0).astype(int)
