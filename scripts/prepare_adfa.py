from pathlib import Path
import pandas as pd

# ADFA-LD: https://github.com/verazuo/a-labelled-version-of-the-ADFA-LD-dataset/blob/master/ADFA-LD%2BSyscall%2BList.txt
ROOT = Path("/home/CIN/rsc8/GraphIDS/data/ADFA-LD")
FIX_HOST = 0
FIX_DST = 1

rows = []
ts = 0

# ==========================
# PROCESSANDO ENTRADAS
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
                # trocar para src e dst para a abordagem system_calls como nós do grafo
                "IPV4_SRC_ADDR": FIX_HOST, # src
                "IPV4_DST_ADDR": FIX_DST, # dst

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
# ARQUIVOS DE TREINAMENTO - BENIGNO
# ==========================

print("Processando arquivos de treinamento")

for file in (ROOT / "Training_Data_Master").glob("*.txt"):

    seq = file.read_text().strip().split()

    process_sequence(
        seq=seq,
        label=0,
        attack_name="normal",
    )


# ==========================
# ARQUIVOS DE VALIDAÇÃO - BENIGNO
# ==========================

print("Processando arquivos de validação")

for file in (ROOT / "Validation_Data_Master").glob("*.txt"):

    seq = file.read_text().strip().split()

    process_sequence(
        seq=seq,
        label=0,
        attack_name="normal",
    )


# ==========================
# ATAQUES
# ==========================

print("Processando arquivos de ataque")

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

# MUDAR CAMINHO DE SAÍDA
outdir = Path(
    "/home/CIN/rsc8/GraphIDS/data/ADFA-LD-h2h"
)

outdir.mkdir(parents=True, exist_ok=True)

outfile = outdir / "ADFA-LD-h2h.csv"

df.to_csv(outfile, index=False)

print("\ndataset criado")
print(outfile)

print("\nshape:")
print(df.shape)

print("\nlabels:")
print(df["Label"].value_counts())

print("\nataques:")
print(df["Attack"].value_counts().head(20))