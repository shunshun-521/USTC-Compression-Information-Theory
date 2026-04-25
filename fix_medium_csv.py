import csv
# 备份现有数据
import shutil, os
if os.path.exists("results_GPTzip-medium.csv.bak2"):
    pass  # 已有备份，不覆盖
else:
    shutil.copy("results_GPTzip-medium.csv", "results_GPTzip-medium.csv.bak2")

# 重新完整写入：包含 1KB / 10KB / 100KB / 1MB(skipped) 四行
with open("results_GPTzip-medium.csv","w",newline="",encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["method","slice","raw_B","comp_B","BPB","t_compress_s","t_decompress_s","throughput_MBps","lossless"])
    w.writerow(["GPTzip-medium","1KB",   1024,    184,   1.4375, 25.1046,  22.115,   4.1e-05,  "通过"])
    w.writerow(["GPTzip-medium","10KB",  10240,   2136,  1.6687, 89.5638,  43.5495,  0.000114, "通过"])
    w.writerow(["GPTzip-medium","100KB", 102400,  20358, 1.5905, 280.5365, 247.7947, 0.000365, "通过(分块)"])
    w.writerow(["GPTzip-medium","1MB",   1048576, "",    "",     "",       "",       "",       "未跑(显存限制)"])

print("已重写 results_GPTzip-medium.csv:")
with open("results_GPTzip-medium.csv","r",encoding="utf-8-sig") as f:
    print(f.read())
