# Guia Passo a Passo: Treinamento e Avaliação do GraphIDS

Este guia cobre desde a preparação dos dados até a avaliação dos resultados, incluindo modo batch (original), modo streaming (novo) e busca de hiperparâmetros.

---

## Índice

1. [Pré-requisitos](#1-pré-requisitos)
2. [Preparação dos Dados](#2-preparação-dos-dados)
3. [Treinamento Batch (Modo Original)](#3-treinamento-batch-modo-original)
4. [Treinamento Streaming (Modo Novo)](#4-treinamento-streaming-modo-novo)
5. [Pipeline Completo: Batch + Streaming](#5-pipeline-completo-batch--streaming)
6. [Runner com Status e Métricas Periódicas](#6-runner-com-status-e-métricas-periódicas)
7. [Busca de Hiperparâmetros](#7-busca-de-hiperparâmetros)
8. [Interpretação dos Resultados](#8-interpretação-dos-resultados)
9. [Apêndice: Comandos Úteis](#9-apêndice-comandos-úteis)

---

## 1. Pré-requisitos

### 1.1 Ambiente

O projeto usa **Python 3.11** e **uv** como gerenciador de pacotes:

```bash
# Verificar instalação
uv --version
python3.11 --version

# Criar ambiente e instalar dependências
uv sync --python 3.11

# IMPORTANTE: Sempre usar --frozen nos comandos, pois as
# dependências pyg-lib e torch-sparse foram removidas em
# favor de um fallback Python puro (patch_pyg.py).
```

### 1.2 Dependências Principais

- `torch` (>=2.1)
- `torch-geometric`
- `pandas`, `numpy`
- `scikit-learn`
- `wandb` (Weights & Biases, modo offline por padrão)
- `patch_pyg.py` → fallback Python puro para NeighborSampler do PyG

> **GLIBC**: O `patch_pyg.py` substitui `pyg-lib` e `torch-sparse` (que exigem GLIBC >= 2.32) por uma implementação pura PyTorch. Funciona em Ubuntu 20.04 (GLIBC 2.31) e sistemas similares.

### 1.3 GPU (Recomendado)

```bash
uv run --python 3.11 --frozen python -c "import torch; print(torch.cuda.is_available())"
# Deve retornar True para usar GPU
```

---

## 2. Preparação dos Dados

### 2.1 Estrutura de Diretórios

```
data/
├── NF-UNSW-NB15-v3/
│   └── NF-UNSW-NB15-v3.csv
├── NF-CSE-CIC-IDS2018-v3/
│   └── NF-CSE-CIC-IDS2018-v3.csv
├── NF-ToN-IoT-v3/
│   └── NF-ToN-IoT-v3.csv
└── ...
```

### 2.2 Datasets Suportados

| Dataset | Arquivo | Modo Batch | Modo Streaming |
|---------|---------|:----------:|:--------------:|
| NF-UNSW-NB15-v3 | `NF-UNSW-NB15-v3.csv` | ✅ | ✅ |
| NF-CSE-CIC-IDS2018-v3 | `NF-CSE-CIC-IDS2018-v3.csv` | ✅ | ✅ |
| NF-ToN-IoT-v3 | `NF-ToN-IoT-v3.csv` | ✅ | ✅ |
| NF-UNSW-NB15-v2 | `NF-UNSW-NB15-v2.csv` | ✅ | ❌ (sem timestamp) |
| NF-CSE-CIC-IDS2018-v2 | `NF-CSE-CIC-IDS2018-v2.csv` | ✅ | ❌ (sem timestamp) |
| ADFA-LD-GraphIDS | `ADFA-LD-GraphIDS.csv` | ✅ | ❌ |
| ADFA-LD-h2h | `ADFA-LD-h2h.csv` | ✅ | ❌ |

> **Nota**: O modo streaming requer datasets `v3` que contêm a coluna `FLOW_START_MILLISECONDS`.

### 2.3 Formato Esperado do CSV

Para datasets **v3**, as colunas relevantes são:

```
IPV4_SRC_ADDR, IPV4_DST_ADDR, FLOW_START_MILLISECONDS,
FLOW_END_MILLISECONDS, L7_PROTO, IN_BYTES, OUT_BYTES, ...,
Attack, Label
```

O Label deve ser 0 (benigno) ou 1 (malicioso).

---

## 3. Treinamento Batch (Modo Original)

### 3.1 Comando Básico

```bash
uv run --python 3.11 --frozen python main.py \
  --dataset NF-UNSW-NB15-v3 \
  --data_dir ./data \
  --split_mode temporal \
  --num_epochs 100
```

### 3.2 Argumentos Principais

| Argumento | Default | Descrição |
|-----------|---------|-----------|
| `--dataset` | NF-UNSW-NB15-v3 | Nome do dataset |
| `--data_dir` | (obrigatório) | Diretório raiz dos dados |
| `--split_mode` | stratified | `stratified`, `temporal`, ou `temporal_shift_aware` |
| `--num_epochs` | 100 | Número de épocas de treino |
| `--batch_size` | 16384 | Batch size do GNN encoder |
| `--ae_batch_size` | 64 | Batch size do Transformer AE |
| `--learning_rate` | 5e-4 | Learning rate |
| `--edim_out` | 64 | Dimensão de saída do encoder |
| `--ae_embedding_dim` | 32 | Dimensão do embedding do Transformer |
| `--window_size` | 512 | Tamanho da janela do Transformer |
| `--seed` | 24 | Seed para reprodutibilidade |
| `--wandb` | false | Habilitar wandb online |
| `--fraction` | None | Fração do dataset para treino (ex.: 0.1) |

### 3.3 Split Modes

#### Stratified (original do artigo)
```bash
uv run --python 3.11 --frozen python main.py \
  --dataset NF-UNSW-NB15-v3 --data_dir ./data \
  --split_mode stratified --num_epochs 100
```
Divisão aleatória mantendo proporção de classes (80/10/10).

#### Temporal
```bash
uv run --python 3.11 --frozen python main.py \
  --dataset NF-UNSW-NB15-v3 --data_dir ./data \
  --split_mode temporal --num_epochs 100
```
Divisão cronológica (80% primeiros fluxos / 10% / 10%).

#### Temporal Shift-Aware
```bash
uv run --python 3.11 --frozen python main.py \
  --dataset NF-CSE-CIC-IDS2018-v3 --data_dir ./data \
  --split_mode temporal_shift_aware --num_epochs 100
```
Divisão que respeita mudanças de distribuição conhecidas (shift days). Dataset `NF-CSE-CIC-IDS2018-v3` apenas.

### 3.4 Exemplo com Métricas

```bash
uv run --python 3.11 python main.py \
  --dataset NF-UNSW-NB15-v3 --data_dir ./data \
  --split_mode temporal --num_epochs 50 \
  --batch_size 16384 --ae_batch_size 64 \
  --learning_rate 5e-4 --edim_out 32 \
  --ae_embedding_dim 16 --window_size 256 \
  --seed 42
```

**Saída esperada ao final:**
```
Test macro F1-score: 0.xxxx
Test PR-AUC: 0.xxxx
Test prediction time: 0.xxxx seconds
```

---

## 4. Treinamento Streaming (Modo Novo)

### 4.1 Comando Básico

```bash
uv run --python 3.11 python main.py \
  --dataset NF-UNSW-NB15-v3 \
  --data_dir ./data \
  --streaming \
  --num_epochs 50
```

Apenas datasets **v3** (com timestamp) são suportados.

### 4.2 O que Acontece

1. **Carregamento**: `StreamingNetFlowDataset` processa o CSV e gera N janelas temporais de 1 hora com stride de 30 minutos.
2. **Fase 1 - Treino** (primeiras 60% janelas): O modelo treina janela por janela. Cada janela é treinada por `max(5, num_epochs // num_windows)` épocas.
3. **Fase 2 - Avaliação** (40% restantes): Para cada janela:
   - Calcula erros de reconstrução via `validate()`
   - Atualiza threshold adaptativo via `update_threshold_online()`
   - Loga F1-score, PR-AUC e threshold para cada janela

### 4.3 Logging (wandb)

Por padrão, o wandb opera em **modo offline** (sem necessidade de conta). Para visualizar:

```bash
# Após a execução, sincronizar:
wandb sync wandb/offline-<run_id>
```

Ou para logging online:
```bash
wandb login
uv run --python 3.11 python main.py \
  --dataset NF-UNSW-NB15-v3 --data_dir ./data \
  --streaming --wandb
```

### 4.4 Exemplo Completo

```bash
uv run --python 3.11 python main.py \
  --dataset NF-UNSW-NB15-v3 --data_dir ./data \
  --streaming --num_epochs 30 \
  --learning_rate 1e-3 --ae_embedding_dim 32 \
  --edim_out 16 --num_layers 2 \
  --dropout 0.1 --mask_ratio 0.2 \
  --window_size 128 --batch_size 16384 \
  --ae_batch_size 64 --step_percent 0.5 \
  --seed 42
```

---

## 5. Pipeline Completo: Batch + Streaming

### 5.1 Usando o Script Automatizado

```bash
./pipelines/train.sh <dataset_name> [options]
```

Exemplo:
```bash
./pipelines/train.sh NF-UNSW-NB15-v3 --data_dir ./data --num_epochs 50
```

O script:
1. Executa modo batch (`--split_mode temporal`)
2. Executa modo streaming (`--streaming`)
3. Extrai F1-score e PR-AUC de ambos
4. Exibe tabela comparativa

Flags:
| Flag | Default | Descrição |
|------|---------|-----------|
| `--data_dir` | ./data | Diretório dos dados |
| `--num_epochs` | 50 | Épocas de treino |
| `--gpu` | auto | Forçar uso de GPU |
| `--dry-run` | - | Apenas mostrar comandos |

### 5.2 Saída do Pipeline

```
========================================
  GraphIDS Training Pipeline
========================================
  Dataset:       NF-UNSW-NB15-v3
  Data dir:      ./data
  Num epochs:    50
  GPU available: True
========================================

--- Step 1: Batch mode training ---
...
Test macro F1-score: 0.8742
Test PR-AUC: 0.9123
Test prediction time: 0.4521 seconds

--- Step 2: Streaming mode training ---
...
stream_window_15_f1: 0.8456
stream_window_15_pr_auc: 0.8901

========================================
  Results Comparison
========================================
Mode                      F1-Score        PR-AUC
------------------------ --------------- ---------------
Batch (temporal split)    0.8742          0.9123
Streaming (temporal)      0.8456          0.8901
Batch test prediction time: 0.4521s
========================================
```

> **Nota**: É esperado que o streaming tenha performance ligeiramente inferior ao batch, pois opera com menos dados por janela e sem épocas completas. A vantagem está na capacidade de inferência contínua e adaptação a mudanças de distribuição.

---

## 6. Runner com Status e Métricas Periódicas

O `run_training.py` é o **runner recomendado** para treino. Ele:
- Escreve progresso em `training_status.json` (consultável via `cat`)
- Mostra métricas (val_pr_auc, best_val_pr_auc) a cada época
- Salva F1-score e PR-AUC final ao terminar

### 6.1 Modo Batch

```bash
uv run --python 3.11 --frozen python run_training.py batch
```

### 6.2 Modo Streaming

```bash
uv run --python 3.11 --frozen python run_training.py streaming
```

### 6.3 Ambos (sequencial)

```bash
uv run --python 3.11 --frozen python run_training.py both
```

### 6.4 Acompanhando o Progresso

**Em outro terminal:**
```bash
# Status atual (JSON formatado)
cat training_status.json

# Logs em tempo real
tail -f training_runner.log

# Extrair métricas rapidamente
grep "EPOCH:" training_runner.log | tail -5
grep "FINAL_TEST_" training_runner.log
```

### 6.5 Parâmetros do Runner (hardcoded)

O `run_training.py` usa os melhores parâmetros encontrados no projeto, ajustados para CPU:

| Parâmetro | Valor | Motivo |
|-----------|-------|--------|
| `--fraction` | 0.3 (30%) | Bom custo-benefício memória/precisão |
| `--batch_size` | 8192 | Reduzido para CPU (vs 16384 padrão) |
| `--ae_batch_size` | 32 | Reduzido para CPU (vs 64 padrão) |
| `--num_epochs` | 30 | Suficiente para convergência em CPU |
| `--learning_rate` | 0.0001 | Do config ótimo do projeto |
| `--edim_out` | 96 | Do config ótimo do projeto |
| `--ae_embedding_dim` | 48 | Do config ótimo do projeto |

Para alterar, edite a lista `BASE_ARGS` em `run_training.py`.

---

## 7. Busca de Hiperparâmetros

### 6.1 Usando o Script Automatizado

```bash
./pipelines/tune.sh <dataset_name> [options]
```

Exemplo:
```bash
./pipelines/tune.sh NF-UNSW-NB15-v3 --data_dir ./data --trials 30 --final
```

Flags:
| Flag | Default | Descrição |
|------|---------|-----------|
| `--trials` | 20 | Número de trials aleatórios |
| `--tune_space` | `config_search_space/tuning_space.yaml` | Arquivo de espaço de busca |
| `--data_dir` | ./data | Diretório dos dados |
| `--final` | false | Rodar treino final com melhores params |
| `--dry-run` | - | Apenas mostrar comandos |

### 6.2 Customizando o Espaço de Busca

#### Para modo batch:
Edite `config_search_space/tuning_space.yaml`:
```yaml
learning_rate:
  type: loguniform
  min: 1e-4
  max: 3e-3
ae_embedding_dim:
  type: choice
  values: [16, 32, 64, 128]
num_layers:
  type: int
  min: 1
  max: 3
mask_ratio:
  type: choice
  values: [0.0, 0.1, 0.2, 0.3, 0.5]
window_size:
  type: choice
  values: [128, 256, 512, 1024]
```

#### Para modo streaming:
Edite `config_search_space/streaming_tuning_space.yaml`:
```yaml
window_size:
  type: choice
  values: [64, 128, 256]
batch_size:
  type: choice
  values: [8192, 16384, 32768]
step_percent:
  type: uniform
  min: 0.25
  max: 1.0
```

### 6.3 Comando Manual de Tuning

```bash
uv run --python 3.11 python main.py \
  --dataset NF-UNSW-NB15-v3 --data_dir ./data \
  --tune \
  --tune_space config_search_space/tuning_space.yaml \
  --tune_trials 20 \
  --tune_num_epochs 30 \
  --tune_patience 10
```

**Saída:**
```
[tuning] trial=1 val_pr_auc=0.8732 params={'learning_rate': 0.00052, ...}
[tuning] trial=2 val_pr_auc=0.8910 params={'learning_rate': 0.00123, ...}
...
[tuning] best_overrides={'learning_rate': 0.00123, 'window_size': 256, ...} best_score=0.9210
```

Os melhores parâmetros são então usados para um treino completo com avaliação no test set.

---

## 8. Interpretação dos Resultados

### 8.1 Métricas Principais

| Métrica | Descrição | Valor Ideal |
|---------|-----------|-------------|
| **F1-score (macro)** | Média harmônica entre precisão e recall | > 0.85 |
| **PR-AUC** | Área sob a curva Precision-Recall | > 0.90 |
| **Prediction time** | Tempo total de inferência no test set | Quanto menor, melhor |
| **Peak GPU memory** | Pico de uso de memória GPU | < 8 GB (depende do batch) |

### 8.2 Comparação Batch vs Streaming

```
Métrica               Batch   Streaming   Diferença
F1-score              0.8742   0.8456     -0.0286 (-3.3%)
PR-AUC                0.9123   0.8901     -0.0222 (-2.4%)
```

Uma diferença de 2-5% é aceitável e esperada. O streaming oferece em troca:
- Inferência contínua flow-a-flow
- Adaptação automática a mudanças de tráfego
- Sem necessidade de retreinar o modelo completo

### 8.3 Threshold Adaptativo

No modo streaming, o threshold evolui ao longo das janelas:
- **Janelas iniciais**: threshold conservador (alto), baixa taxa de alertas
- **Janelas intermediárias**: threshold se ajusta com o histórico de erros
- **Janelas finais**: threshold estabiliza em um valor ótimo para a distribuição corrente

### 8.4 Logs do wandb

Com wandb offline, os logs ficam em `wandb/offline-<run_id>/`. Para visualizar:

```bash
# Listar runs offline
ls wandb/offline-*/

# Sincronizar com servidor (requer login)
wandb sync wandb/offline-<run_id>/
```

---

## 9. Apêndice: Comandos Úteis

### 9.1 Testes

```bash
# Executar todos os testes
uv run --python 3.11 python -m pytest tests/ -v

# Testes específicos
uv run --python 3.11 python -m pytest tests/test_dynamic_graph.py -v
uv run --python 3.11 python -m pytest tests/test_streaming.py -v
uv run --python 3.11 python -m pytest tests/test_model_adaptations.py -v

# Com cobertura
uv run --python 3.11 python -m pytest tests/ --cov=. --cov-report=term
```

### 9.2 Reprocessar Dataset

```bash
# Forçar reprocessamento com nova seed
uv run --python 3.11 python main.py \
  --dataset NF-UNSW-NB15-v3 --data_dir ./data \
  --reload_dataset --seed 99
```

### 9.3 Usar Fração do Dataset (para testes rápidos)

```bash
uv run --python 3.11 python main.py \
  --dataset NF-UNSW-NB15-v3 --data_dir ./data \
  --fraction 0.1 --num_epochs 10
```

### 9.4 Dry-run dos Scripts de Pipeline

```bash
# Ver comandos sem executar
./pipelines/train.sh NF-UNSW-NB15-v3 --dry-run
./pipelines/tune.sh NF-UNSW-NB15-v3 --dry-run
```

### 9.5 Modo Debug / Verbose

O wandb offline já loga todas as métricas. Para verbose adicional, o próprio stdout do `main.py` exibe:
- Configuração completa do run
- Loss de treino e validação a cada época
- Métricas finais de teste

### 9.6 Checkpoints

Os checkpoints são salvos em `checkpoints/GraphIDS_<dataset>_<seed>.ckpt` (batch) ou `checkpoints/GraphIDS_<dataset>_<seed>_trial<N>.ckpt` (tuning). Para carregar:

```python
from models.graphids import GraphIDS
model = GraphIDS(...)
model.load_checkpoint("checkpoints/GraphIDS_NF-UNSW-NB15-v3_42.ckpt")
```

### 9.7 Resumo Rápido

```bash
# 0. IMPORTANTE: Sempre usar --frozen com uv run
#    (pyg-lib e torch-sparse substituídos por patch_pyg.py)

# 1. Treino batch (runner recomendado)
uv run --python 3.11 --frozen python run_training.py batch

# 2. Treino streaming
uv run --python 3.11 --frozen python run_training.py streaming

# 3. Batch + Streaming sequencial
uv run --python 3.11 --frozen python run_training.py both

# 4. Acompanhar progresso
cat training_status.json
tail -f training_runner.log

# 5. Busca de hiperparâmetros
uv run --python 3.11 --frozen python main.py \
  --dataset NF-UNSW-NB15-v3 --data_dir ./data \
  --tune --tune_space config_search_space/tuning_space.yaml \
  --tune_trials 20
```
