import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
FILES = [
    BASE / "Video Watching and Reflection Study Form 1 (Responses).xlsx",
    BASE / "_ENG3530 Video Watching and Reflection (Responses).xlsx",
]

for f in FILES:
    print("=" * 80)
    print(f.name)
    xl = pd.ExcelFile(f)
    print("sheets:", xl.sheet_names)
    for s in xl.sheet_names:
        df = xl.parse(s)
        print(f"--- sheet '{s}': {df.shape[0]} rows x {df.shape[1]} cols")
        for c in df.columns:
            print("   COL:", repr(c))
