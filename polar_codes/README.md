# 极化码编译码仿真

Python（NumPy/SciPy）实现的极化码 GA 构造、SC/SCL/CA-SCL/BP 译码与蒙特卡洛仿真。

## 依赖

```bash
pip install -r requirements.txt
```

## 校验

```bash
cd polar_codes
python3 validate.py
```

编码器自检：`u=[1,0,1,1]` → `x=[1,0,1,1]`（`x = u @ B_N @ F^{\otimes n}` mod 2）。

## 实验

```bash
python3 run_exp1.py
python3 run_exp2.py
python3 run_exp3.py
```

快速冒烟（缩短 SNR 网格与帧数）：

```bash
export POLAR_MAX_FRAMES=3000 POLAR_MIN_ERRORS=30
export POLAR_EB_MIN=2.0 POLAR_EB_MAX=4.0 POLAR_EB_STEP=0.5
export POLAR_SKIP_N1024=1 POLAR_N256_ONLY=1
python3 run_exp1.py && python3 run_exp2.py && python3 run_exp3.py
```

结果写入 `results/`（CSV、PNG/PDF、`frozen_sets.txt`）。
