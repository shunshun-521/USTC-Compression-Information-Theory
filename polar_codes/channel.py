"""
AWGN 信道 + BPSK 调制/解调

实现与极化码 SCD 译码器配套：0 -> -sqrt(Ec), 1 -> +sqrt(Ec)，
LLR = ln P(y|0)/P(y|1) = -2 y sqrt(Ec) / N0（N0=1）。
与常见 0->+1 约定相差整体符号翻转，Eb/N0 与 BLER 曲线等价。
"""
import numpy as np


def _code_energy(rate):
    """每比特能量归一化因子 sqrt(Ec)，Ec = 2R Eb。"""
    return np.sqrt(2.0 * rate)


def bpsk_modulate(x, rate=0.5):
    """将二进制码字 x (0/1) 映射为 BPSK 符号。"""
    x = np.asarray(x, dtype=np.float64)
    es = _code_energy(rate)
    return (2.0 * x - 1.0) * es


def awgn_channel(s, sigma, rng=None):
    """加高斯白噪声，返回接收信号 y = s + n，n ~ N(0, sigma^2)。"""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=np.shape(s))
    return s + noise


def compute_llr(y, sigma, rate=0.5):
    """
    信道 LLR（与 SCD 一致）。
    采用 N0=1，sigma 为噪声标准差，Ec = 2R。
    """
    es = _code_energy(rate)
    n0 = sigma ** 2
    return -2.0 * np.asarray(y, dtype=np.float64) * es / n0


def eb_n0_to_sigma(eb_n0_db, rate):
    """
    将 Eb/N0 (dB) 转换为 AWGN 噪声标准差 sigma。
    SNR = Eb/N0 * 2R（线性），N0=1 => sigma = 1/sqrt(SNR)。
    """
    snr_linear = (10.0 ** (eb_n0_db / 10.0)) * 2.0 * rate
    return 1.0 / np.sqrt(snr_linear)
