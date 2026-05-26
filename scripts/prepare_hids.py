import pandas as pd
import pyarrow.parquet as pq

pf = pq.ParquetFile("process_uber_summary.parquet")

target = 1_000_000
batches = []
read = 0

for rg in range(pf.num_row_groups):
    table = pf.read_row_group(rg)  # pode passar columns=[...]
    batches.append(table)
    read += table.num_rows
    if read >= target:
        break

df = table = None
df = pd.concat([t.to_pandas() for t in batches], ignore_index=True).head(target)
df.to_csv("process_uber_summary.csv", index=False)
