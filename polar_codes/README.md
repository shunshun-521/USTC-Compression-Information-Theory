# 极化码（Polar Codes）编译码仿真

纯 Python（NumPy/SciPy）实现，无第三方极化码库。

## 目录

| 文件 | 说明 |
|------|------|
| `construction.py` | GA 高斯近似构造 |
| `encoder.py` | O(N log N) 蝶形编码 |
| `decoder_sc.py` | SC 译码（非递归 + 递归参考） |
| `decoder_scl.py` | SCL / CA-SCL（CRC-8/16） |
| `decoder_bp.py` | BP（min-sum + 早停） |
| `channel.py` | BPSK + AWGN |
| `simulation.py` | 蒙特卡洛仿真 |
| `utils.py` | CSV、绘图、香农限 |
| `run_exp1.py` ~ `run_exp3.py` | 三组实验 |

## 运行

```bash
cd polar_codes
pip install -r requirements.txt
python construction.py   # 打印构造校验
python run_exp1.py       # SC 仿真
python run_exp2.py       # SCL / CA-SCL
python run_exp3.py       # BP 对比
```

快速验证（减少帧数与 SNR 点数）：

```bash
POLAR_QUICK=1 python run_exp1.py
```

## 单元测试

- 编码：`polar_encode([1,0,1,1])` → `[1,1,0,1]`
- SC：高 SNR 下 `verify_sc_decoders`
- SCL L=1 等价 SC：`verify_scl_equals_sc`

## 结果

输出在 `results/`：CSV、`frozen_sets.txt`、PNG/PDF 曲线图。
