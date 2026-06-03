# 极化码编译码仿真

Python（NumPy/SciPy）实现的极化码 GA 构造、SC/SCL/CA-SCL/BP 译码与蒙特卡洛仿真。

## 目录

- `construction.py` — 高斯近似（GA）构造
- `encoder.py` — 蝶形编码
- `decoder_sc.py` — SC 译码（层化 LLR）
- `decoder_scl.py` — SCL / CRC 辅助译码
- `decoder_bp.py` — BP 译码（分层 min-sum + 迭代增强 SC）
- `channel.py` — BPSK-AWGN
- `simulation.py` — 蒙特卡洛主循环
- `utils.py` — CSV、绘图、容量限
- `run_exp1.py` / `run_exp2.py` / `run_exp3.py` — 实验脚本

## 运行

```bash
cd polar_codes
pip install -r requirements.txt
python construction.py          # GA 校验
python run_exp1.py              # SC 仿真
python run_exp2.py              # SCL 仿真
python run_exp3.py              # BP 对比
```

快速模式（较少帧数/信噪比点，便于 CI）：

```bash
export POLAR_FAST=1
python run_exp1.py && python run_exp2.py && python run_exp3.py
```

完整仿真请去掉 `POLAR_FAST`，并预留较长运行时间。

## 编码器校验

`u=[1,0,1,1]` → `x=[1,1,0,1]`（`x = u * G_N`）。
