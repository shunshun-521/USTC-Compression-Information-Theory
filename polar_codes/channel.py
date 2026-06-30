"""
AWGN 信道 + BPSK 调制/解调
调制：0 -> +sqrt(Es), 1 -> -sqrt(Es)，Es = 2R * 10^{Eb/N0/10}
"""
import numpy as np


def snr_linear(eb_n0_db, rate):
    """线性信噪比 SNR = 2R * 10^{Eb/N0/10}。"""
    return 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))


def bpsk_modulate(x, energy=None):
    """
    将二进制码字 x (0/1) 映射为 BPSK 符号。
    energy 为符号能量 Es；默认 1（单位能量）。
    """
    symbols = 1.0 - 2.0 * np.asarray(x, dtype=np.float64)
    if energy is not None:
        symbols = symbols * np.sqrt(float(energy))
    return symbols


def awgn_channel(s, sigma, rng=None):
    """加高斯白噪声，返回接收信号 y = s + n，n ~ N(0, sigma^2)"""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=np.shape(s))
    return s + noise


def compute_llr(y, sigma):
    """
    计算 BPSK-AWGN 信道的信道 LLR。
    LLR(y) = ln P(y|x=0) / P(y|x=1) = 2*y / sigma^2
    """
    return 2.0 * np.asarray(y, dtype=np.float64) / (sigma ** 2)


def eb_n0_to_sigma(eb_n0_db, rate):
    """
    将 Eb/N0 (dB) 转换为 AWGN 噪声标准差 sigma。
    SNR = Eb/N0 * 2R（线性）
    sigma = 1 / sqrt(SNR) = 1 / sqrt(2R * 10^{Eb/N0/10})
    """
    return 1.0 / np.sqrt(snr_linear(eb_n0_db, rate))
