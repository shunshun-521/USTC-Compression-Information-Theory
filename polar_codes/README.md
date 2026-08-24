# 极化码编译码仿真

Python 实现的极化码（Polar Codes）完整仿真框架，包含 GA 构造、SC/SCL/CA-SCL/BP 译码与蒙特卡洛仿真。

## 快速开始

```bash
cd polar_codes
pip install -r requirements.txt
python3 validate.py          # 单元测试
python3 run_exp1.py          # SC 仿真
python3 run_exp2.py          # SCL / CA-SCL 仿真
python3 run_exp3.py          # BP 仿真
```

## 加速仿真（可选）

```bash
POLAR_FAST_SIM=1 POLAR_MAX_FRAMES=5000 POLAR_MIN_ERRORS=20 python3 run_exp1.py
```

## 目录结构

- `construction.py` — 高斯近似（GA）极化码构造
- `encoder.py` — Arikan 蝶形编码器
- `decoder_sc.py` — PSCD 非递归 SC 译码
- `decoder_scl.py` — SCL / CA-SCL 译码（含 CRC-8/16）
- `decoder_bp.py` — BP 译码（scaled min-sum + 早停）
- `channel.py` — BPSK-AWGN 信道
- `simulation.py` — 蒙特卡洛仿真主循环
- `utils.py` — 结果保存、绘图、BPSK 容量限
- `validate.py` — 数值校验
- `results/` — 仿真输出（CSV、PNG、PDF）
