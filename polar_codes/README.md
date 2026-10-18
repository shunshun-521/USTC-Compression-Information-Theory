# 极化码编译码仿真（Python）

实现 GA 构造、蝶形编码、AWGN-BPSK 信道、SC / SCL(CA-SCL) / BP 译码与蒙特卡洛仿真。

## 运行

```bash
cd polar_codes
pip install -r requirements.txt
python3 verify.py
python3 run_exp1.py
python3 run_exp2.py
python3 run_exp3.py
```

快速冒烟（较少帧数）：

```bash
POLAR_QUICK=1 python3 run_exp1.py
```

## 说明

- 信道 LLR 在进入译码器前会做比特倒序重排（`channel.reorder_llr_for_decoder`），与编码端 `polar_encode` 的倒序输出一致。
- 编码器自检向量 `u=[1,0,1,1]` 对应码字 `[1,0,1,1]`（与 $G_N=B_N F^{\otimes n}$ 矩阵乘法一致）。
