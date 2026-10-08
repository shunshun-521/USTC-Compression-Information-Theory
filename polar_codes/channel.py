"""
AWGN 信道 + BPSK 调制/解调
调制：0 -> +1, 1 -> -1
"""
import numpy as np


def bpsk_modulate(x):
    """BPSK：0 -> -1，1 -> +1（与 SC 因子图实现一致）"""
    return 2.0 * (np.asarray(x, dtype=np.float64) - 0.5)


def awgn_channel(s, sigma, rng=None):
    """加高斯白噪声，返回接收信号 y = s + n，n ~ N(0, sigma^2)"""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=np.shape(s))
    return s + noise


def compute_llr(y, sigma):
    """
    计算 BPSK-AWGN 信道 LLR（ln P(y|0)/P(y|1)）。
    采用 Es=1、实噪声方差 sigma^2：LLR = -2y / (2 sigma^2) * sigma^2 的等价形式 -2y/No，No=2*sigma^2。
    """
    y = np.asarray(y, dtype=np.float64)
    no = 2.0 * (sigma ** 2)
    return -2.0 * y / no


def eb_n0_to_sigma(eb_n0_db, rate):
    """
    将 Eb/N0 (dB) 转换为 AWGN 噪声标准差 sigma。
    SNR = Eb/N0 * 2R（线性）
    sigma = 1 / sqrt(SNR) = 1 / sqrt(2R * 10^{Eb/N0/10})
    """
    snr = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))
    return 1.0 / np.sqrt(snr)
