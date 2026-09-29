"""
AWGN 信道 + BPSK 调制/解调
调制：0 -> +1, 1 -> -1
"""
import numpy as np


def bpsk_modulate(x):
    """将二进制码字 x (0/1) 映射为 BPSK 符号 (+1/-1)"""
    return 1.0 - 2.0 * np.asarray(x, dtype=np.float64)


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


def reorder_llr_for_decoder(llr, N):
    """极化编码含比特倒序时，将信道 LLR 重排为 SC/SCL 译码器输入顺序"""
    n = int(np.log2(N))
    br = np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)
    inv = np.empty(N, dtype=int)
    inv[br] = np.arange(N)
    return np.asarray(llr, dtype=np.float64)[inv]


def eb_n0_to_sigma(eb_n0_db, rate):
    """
    将 Eb/N0 (dB) 转换为 AWGN 噪声标准差 sigma。
    """
    snr = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))
    return 1.0 / np.sqrt(snr)
