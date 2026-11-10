#!/usr/bin/env python
import importlib, shutil, sys
from pathlib import Path

mods = ["numpy","pandas","scipy","sklearn","statsmodels","pyarrow","duckdb","bed_reader","tqdm","psutil","torch"]
print("Python:", sys.version.replace("\n"," "))
for m in mods:
    try:
        x = importlib.import_module(m)
        print(f"{m:14s} OK   {getattr(x,'__version__','')}")
    except Exception as e:
        print(f"{m:14s} MISSING/ERROR   {e}")

print("plink1.9 :", shutil.which("plink1.9"))

print("twstats  :", shutil.which("twstats"))
print("twtable  : literal argument 'twtable'")

try:
    import torch
    print("torch CUDA available:", torch.cuda.is_available())
    print("torch build:", torch.__version__)
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(i, torch.cuda.get_device_name(i), f"{p.total_memory/1024**3:.1f} GB")
except Exception as e:
    print("Torch GPU check failed:", repr(e))
