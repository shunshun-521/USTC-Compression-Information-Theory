# 极化码编译码仿真

Python（NumPy/SciPy）实现的极化码 GA 构造、SC/SCL/CA-SCL/BP 译码与蒙特卡洛仿真。

## 目录

- `construction.py` — 高斯近似（GA）构造
- `encoder.py` — 蝶形编码（$x = u F^{\otimes n}$）
- `decoder_sc.py` — SC 译码（递归 boxplus）
- `decoder_scl.py` — SCL / CA-SCL（列表译码，含 SC 候选路径）
- `decoder_bp.py` — BP 译码（min-sum + 早停）
- `channel.py` — BPSK-AWGN
- `simulation.py` — 蒙特卡洛主循环
- `utils.py` — CSV、绘图、容量限
- `run_exp1.py` / `run_exp2.py` / `run_exp3.py` — 实验脚本

## 运行

```bash
cd polar_codes
pip install -r requirements.txt
python construction.py    # 验证 GA 构造
python run_exp1.py        # SC 仿真
python run_exp2.py        # SCL / CA-SCL
python run_exp3.py        # BP 对比
```

快速冒烟测试（较少帧数）：

```bash
POLAR_QUICK_TEST=1 python run_exp1.py
```

完整仿真默认 `max_frames=100000`、`min_errors=100`，耗时较长。

## 结果

输出保存在 `results/`：CSV、BLER 曲线图（PNG/PDF）、`frozen_sets.txt`。
