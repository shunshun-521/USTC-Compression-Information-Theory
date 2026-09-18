# 极化码编译码仿真

Python（NumPy/SciPy）实现的极化码 GA 构造、SC/SCL/CA-SCL/BP 译码与蒙特卡洛仿真。

## 目录

- `construction.py` — 高斯近似 / Bhattacharyya 构造
- `encoder.py` — 蝶形编码
- `decoder_sc.py` — SC 译码（非递归主实现）
- `decoder_scl.py` — SCL / CA-SCL
- `decoder_bp.py` — 基于校验矩阵的 min-sum BP
- `simulation.py` — 蒙特卡洛主循环
- `run_exp1.py` / `run_exp2.py` / `run_exp3.py` — 实验脚本

## 运行

```bash
cd polar_codes
pip install -r requirements.txt
python run_exp1.py
python run_exp2.py
python run_exp3.py
```

快速冒烟测试（较少帧数、较粗 SNR 步进）：

```bash
POLAR_QUICK=1 python run_exp1.py
POLAR_QUICK=1 python run_exp2.py
POLAR_QUICK=1 python run_exp3.py
```

完整仿真（`MAX_FRAMES=100000`）在普通 CPU 上可能需要数小时；BP 译码在较大码长下尤其耗时。

## 说明

- 信道 LLR：`LLR = 2y/σ²`，BPSK 映射 `0→+1, 1→-1`
- 编码与 SC/SCL 采用与因子图一致的蝶形结构；SC 按比特倒序相位更新
- BP 在码字域运行 min-sum 后通过 `G⁻¹` 映射回 `u`
