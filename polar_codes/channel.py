"""
AWGN 信道 + BPSK 调制/解调
调制：0 -> -sqrt(Es), 1 -> +sqrt(Es)
LLR = ln P(y|0)/P(y|1) = -2*y*sqrt(Es)/No
"""
import numpy as np


def bpsk_modulate(x, es=1.0):
    """将二进制码字 x (0/1) 映射为 BPSK 符号"""
    x = np.asarray(x, dtype=np.float64)
    return 2.0 * (x - 0.5) * np.sqrt(es)


def awgn_channel(s, sigma, rng=None):
    """加高斯白噪声"""
    s = np.asarray(s, dtype=np.float64)
    if rng is None:
        noise = np.random.normal(0.0, sigma, size=s.shape)
    else:
        noise = rng.normal(0.0, sigma, size=s.shape)
    return s + noise


def compute_llr(y, sigma, es=1.0):
    """
    计算 BPSK-AWGN 信道 LLR。
    与 Es=1, No=sigma^2 一致：LLR = -2*y*sqrt(Es)/No = -2*y/sigma^2
    """
    y = np.asarray(y, dtype=np.float64)
    no = sigma ** 2
    return -2.0 * y * np.sqrt(es) / no


def eb_n0_to_sigma(eb_n0_db, rate, es=1.0):
    """
    Eb/N0 (dB) -> 噪声标准差 sigma。
    Es/N0 = Eb/N0 * R * (1/R) ... SNR_b = Eb/N0 * 2R（每实维度）
    No = Es / (Eb/N0_lin * 2R), sigma = sqrt(No/2) 对实 AWGN 每维
    """
    eb_lin = 10.0 ** (eb_n0_db / 10.0)
    no = es / (eb_lin * 2.0 * rate)
    return np.sqrt(no / 2.0)
