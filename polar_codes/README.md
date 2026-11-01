# 极化码编译码仿真

Python 实现（NumPy/SciPy），无第三方极化码库。

## 安装

```bash
pip install -r polar_codes/requirements.txt
```

## 校验

```bash
cd polar_codes && python3 validate.py
```

## 实验

```bash
cd polar_codes
python3 run_exp1.py
python3 run_exp2.py
python3 run_exp3.py
```

长时间仿真可通过环境变量缩短，例如：

```bash
export POLAR_MAX_FRAMES=5000
export POLAR_MIN_ERRORS=50
export POLAR_SKIP_N1024=1
```

## 说明

- 编码采用蝶形结构 + 比特倒序置换，与 `u @ G_N`（`G_N = B_N F^{\otimes n}`）一致。
- SC/SCL 译码使用 Permuted SC，信道 LLR 在译码前做比特倒序置换。
- GA 构造 N=8,K=4 @2.5dB 信息位索引：`[3, 5, 6, 7]`。
