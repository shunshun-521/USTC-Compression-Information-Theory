import csv
with open("results_qwen.csv","w",newline="",encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["method","slice","raw_B","comp_B","BPB","t_compress_s","t_decompress_s","throughput_MBps","lossless"])
    w.writerow(["Qwen2.5-0.5B","1KB",1024,56,0.4375,24.41,10.57,0.000042,"通过"])
    w.writerow(["Qwen2.5-0.5B","10KB",10240,941,0.7352,2446.64,453.38,0.000004,"通过"])

print("已写入 results_qwen.csv:")
with open("results_qwen.csv","r",encoding="utf-8-sig") as f:
    print(f.read())
