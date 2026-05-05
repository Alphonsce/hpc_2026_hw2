from pathlib import Path
import pandas as pd

PROCESS_COUNTS = (1, 2, 4, 8)

frames = []

for processes in PROCESS_COUNTS:
    path = Path(f"results/ca_speedup_p{processes}.csv")
    frame = pd.read_csv(path)
    frames.append(frame[["processes", "time_seconds"]])

raw = pd.concat(frames, ignore_index=True)
raw.to_csv("results/ca_speedup_raw.csv", index=False)

speedup = raw.copy()
base_time = speedup.loc[0, "time_seconds"]
speedup["speedup"] = base_time / speedup["time_seconds"]
speedup["time_seconds"] = speedup["time_seconds"].map(lambda value: f"{value:.6f}")
speedup["speedup"] = speedup["speedup"].map(lambda value: f"{value:.6f}")
speedup.to_csv("results/ca_speedup.csv", index=False)