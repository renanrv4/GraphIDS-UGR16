from pathlib import Path

import pandas as pd

# ADFA-LD: https://github.com/verazuo/a-labelled-version-of-the-ADFA-LD-dataset/blob/master/ADFA-LD%2BSyscall%2BList.txt
ROOT = Path("/home/CIN/rsc8/GraphIDS/data/ADFA-LD")
FIX_HOST = 0
FIX_DST = 1

rows = []
ts = 0

# ==========================
# PROCESSING ENTRIES
# ==========================


def process_sequence(seq, label, attack_name):
    global ts

    seq = [int(x) for x in seq]

    trace_length = len(seq)

    if trace_length < 2:
        return

    for i in range(trace_length - 1):
        src = seq[i]
        dst = seq[i + 1]

        prev_syscall = seq[i - 1] if i > 0 else -1
        next_syscall = seq[i + 2] if i < trace_length - 2 else -1

        rows.append(
            {
                "FLOW_START_MILLISECONDS": ts,
                # change to src and dst IPs to use system_calls as nodes to GNN, since ADFA-LD is a single host dataset
                "IPV4_SRC_ADDR": FIX_HOST,  # src - source ip
                "IPV4_DST_ADDR": FIX_DST,  # dst - destination ip
                # edge features
                "src_syscall": src,
                "dst_syscall": dst,
                "prev_syscall": prev_syscall,
                "next_syscall": next_syscall,
                "position": i,
                "trace_length": trace_length,
                "count": 1,
                "Label": label,
                "Attack": attack_name,
            }
        )

        ts += 1


# ==========================
# TRAINING FILES - BENIGN
# ==========================

print("Processing training files")

for file in (ROOT / "Training_Data_Master").glob("*.txt"):
    seq = file.read_text().strip().split()

    process_sequence(
        seq=seq,
        label=0,
        attack_name="normal",
    )


# ==========================
# VALIDATION FILES - BENIGN
# ==========================

print("Processing validation files")

for file in (ROOT / "Validation_Data_Master").glob("*.txt"):
    seq = file.read_text().strip().split()

    process_sequence(
        seq=seq,
        label=0,
        attack_name="normal",
    )


# ==========================
# ATTACK FILES - MALICIOUS
# ==========================

print("Processing attack files")

for file in (ROOT / "Attack_Data_Master").rglob("*.txt"):
    attack_type = file.parent.name

    seq = file.read_text().strip().split()

    process_sequence(
        seq=seq,
        label=1,
        attack_name=attack_type,
    )


# ==========================
# SAVE
# ==========================

df = pd.DataFrame(rows)

# CHANGE OUTPUT PATH BELOW TO YOUR OWN
outdir = Path("/home/CIN/rsc8/GraphIDS/data/ADFA-LD-h2h")

outdir.mkdir(parents=True, exist_ok=True)

# CSV FILE WITH ALL PROCESSED DATA
outfile = outdir / "ADFA-LD-h2h.csv"

df.to_csv(outfile, index=False)

print("\nSaved processed dataset to:")
print(outfile)

print("\nshape:")
print(df.shape)

print("\nlabels:")
print(df["Label"].value_counts())

print("\nattacks:")
print(df["Attack"].value_counts().head(20))
