"""
AWGN 信道 + BPSK 调制/解调
调制：0 -> -sqrt(Es), 1 -> +sqrt(Es)（与标准 polar-codes 库一致）
LLR = ln P(y|0)/P(y|1) = -2*y*sqrt(Es)/No
"""
import numpy as np


def bpsk_modulate(x, Es=1.0):
    """将二进制码字 x (0/1) 映射为 BPSK 符号"""
    x = np.asarray(x)
    return 2.0 * (x - 0.5) * np.sqrt(Es)


def awgn_channel(s, sigma, rng=None):
    """
    加高斯白噪声。
    sigma 为噪声标准差（每维），对应 No/2 = sigma^2。
    """
    s = np.asarray(s)
    if rng is None:
        noise = np.random.normal(0, sigma, size=s.shape)
    else:
        noise = rng.normal(0, sigma, size=s.shape)
    return s + noise


def compute_llr(y, Es=1.0, No=None, sigma=None):
    """
    计算 BPSK-AWGN 信道的信道 LLR。
    LLR(y) = -2*y*sqrt(Es)/No
    """
    if No is None:
        if sigma is None:
            raise ValueError("Either No or sigma must be provided")
        No = 2.0 * sigma ** 2
    return -2.0 * y * np.sqrt(Es) / No


def eb_n0_to_es(eb_n0_db, rate):
    """Es/No = (Eb/No) * R（线性），返回 Es（No=1）"""
    eb_no = 10 ** (eb_n0_db / 10.0)
    return eb_no * rate


def eb_n0_to_sigma(eb_n0_db, rate):
    """
    噪声标准差 sigma = sqrt(No/2)，No=1 固定（与 polar-codes 归一化一致）。
    """
    no = 1.0
    return np.sqrt(no / 2.0)


def channel_params(eb_n0_db, rate):
    """返回 (Es, No, sigma) 用于一次仿真点"""
    es = eb_n0_to_es(eb_n0_db, rate)
    no = 1.0
    sigma = np.sqrt(no / 2.0)
    return es, no, sigma
