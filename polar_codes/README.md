# 极化码编译码仿真

基于 NumPy/SciPy 的极化码实现（无第三方极化码库），包含 GA 构造、SC/SCL/CA-SCL/BP 译码与蒙特卡洛仿真。

## 运行

```bash
cd polar_codes
pip install -r requirements.txt
python run_exp1.py   # SC
python run_exp2.py   # SCL / CA-SCL
python run_exp3.py   # BP 对比
```

快速冒烟测试（减少帧数）：

```bash
POLAR_QUICK=1 python run_exp1.py
```

## 目录

- `construction.py` — 高斯近似 (GA) 构造
- `encoder.py` / `decoder_*.py` — 编解码
- `channel.py` / `simulation.py` — 信道与仿真
- `results/` — CSV、曲线图与 `frozen_sets.txt`
