# 极化码编译码仿真（Python / NumPy）

实现 GA 构造、蝶形编码、Permuted SC/SCL/BP 译码与蒙特卡洛仿真。

## 依赖

```bash
pip install -r requirements.txt
```

## 校验

```bash
python3 validate.py
```

## 实验

```bash
python3 run_exp1.py
python3 run_exp2.py
python3 run_exp3.py
```

自动化/快速运行可通过环境变量缩短仿真，例如：

```bash
export POLAR_MAX_FRAMES=2000
export POLAR_MIN_ERRORS=50
export POLAR_EB_N0_LIST=3.0,4.0,5.0,6.0
export POLAR_N256_ONLY=1
export POLAR_SKIP_N1024=1
```

## 说明

- 信道 LLR：`LLR = 2y/σ²`（正号倾向比特 0）
- SC/SCL 采用 Permuted SCD，与分阶段蝶形编码器配套
- 编码器自检：`u=[1,0,1,1]` → `x=[1,1,0,1]`（与 `u @ G_N` 一致）
