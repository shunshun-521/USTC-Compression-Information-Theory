# 极化码编译码仿真

Python 实现的极化码（Polar Codes）完整仿真流水线，不依赖第三方极化码库。

## 模块

| 文件 | 功能 |
|------|------|
| `construction.py` | GA 高斯近似构造 |
| `encoder.py` | 蝶形 O(N log N) 编码 |
| `channel.py` | BPSK + AWGN |
| `decoder_sc.py` | SC 译码（Vangala 非递归 + 递归参考） |
| `decoder_scl.py` | SCL / CA-SCL（CRC-8/16） |
| `decoder_bp.py` | LDPC 风格 min-sum BP + 早停 |
| `simulation.py` | 蒙特卡洛仿真 |
| `utils.py` | CSV 导出、绘图、容量限 |

## 运行

```bash
cd polar_codes
pip install -r requirements.txt
python3 run_exp1.py   # SC 仿真
python3 run_exp2.py   # SCL / CA-SCL 仿真
python3 run_exp3.py   # BP 对比仿真
```

快速模式（减少帧数/SNR 点，适合 CI）：

```bash
POLAR_QUICK=1 python3 run_exp1.py
```

## 约定

- LLR：正号倾向比特 0，`LLR = 2y/σ²`
- 编码：`x = u @ F^⊗n`（蝶形 XOR，无输出比特倒序）
- SC 译码：按比特倒序索引顺序译码（Vangala 风格）
