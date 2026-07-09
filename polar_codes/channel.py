"""
AWGN 信道 + BPSK 调制/解调
调制：0 -> -sqrt(Es), 1 -> +sqrt(Es)，Es = Eb_lin * R
噪声：N(0, 1/2)（No=1 归一化）
"""
import numpy as np


def symbol_energy(eb_n0_db, rate):
    """每符号能量 Es（线性）"""
    return (10.0 ** (eb_n0_db / 10.0)) * rate


def bpsk_modulate(x, eb_n0_db, rate):
    """将二进制码字 x (0/1) 映射为 BPSK 符号"""
    es = symbol_energy(eb_n0_db, rate)
    x = np.asarray(x, dtype=np.float64)
    return (2.0 * x - 1.0) * np.sqrt(es)


def awgn_channel(s, sigma, rng=None):
    """加高斯白噪声，返回接收信号 y = s + n，n ~ N(0, sigma^2)"""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=np.shape(s))
    return s + noise


def compute_llr(y, eb_n0_db, rate, sigma):
    """
    计算 BPSK-AWGN 信道的信道 LLR。
    LLR(y) = ln P(y|x=0) / P(y|x=1) = -2*y*sqrt(Es)/sigma^2
    """
    es = symbol_energy(eb_n0_db, rate)
    y = np.asarray(y, dtype=np.float64)
    return -2.0 * y * np.sqrt(es) / (sigma ** 2)


def eb_n0_to_sigma(eb_n0_db, rate):
    """
    将 Eb/N0 (dB) 转换为 AWGN 噪声标准差 sigma。
    采用 No=1 归一化，sigma = sqrt(No/2) = 1/sqrt(2)，与 Eb/N0 无关。
    """
    _ = eb_n0_db
    _ = rate
    return np.sqrt(0.5)
