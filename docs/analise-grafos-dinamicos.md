# Análise e Plano de Adaptação do GraphIDS para Topologias de Rede Dinâmicas

## 1. Resumo do Problema

O GraphIDS atual constrói um **grafo estático único** a partir de todo o dataset de uma só vez.
Não há suporte para **adição/remoção de nós (IPs)** ou **arestas (fluxos)** em tempo real,
o que o torna inadequado para topologias de rede instáveis onde hosts entram e saem
constantemente (redes móveis, IoT, cloud elástica, etc.).

---

## 2. Limitações Críticas do Código Atual

### 2.1 Construção de Grafo Estática

Em `utils/dataloaders.py:342-353`, o `node_map` é construído **uma única vez** a partir
de todos os IPs únicos de **train + val + test combinados**:

```python
unique_nodes = pd.concat([
    df_train["IPV4_SRC_ADDR"], df_train["IPV4_DST_ADDR"],
    df_val["IPV4_SRC_ADDR"],   df_val["IPV4_DST_ADDR"],
    df_test["IPV4_SRC_ADDR"],  df_test["IPV4_DST_ADDR"],
]).unique()
node_map = {node: i for i, node in enumerate(unique_nodes)}
```

Isso pressupõe que **todos os nós futuros são conhecidos de antemão** — inviável
para cenários com aparecimento de novos IPs.

### 2.2 Features de Nó Triviais

Em `utils/dataloaders.py:365`, as features de nó são um tensor de **1s**:

```python
x = torch.ones(num_nodes, edge_attr.shape[1], dtype=torch.float)
```

Não há representação de nó que evolua com o tempo. Quando um novo nó aparece,
ele começa sem histórico de arestas para gerar uma representação significativa
(**cold-start problem**).

### 2.3 Pipeline Offline em Lote

O fluxo inteiro (`_process()`) é:
1. Ler CSV completo
2. Dividir em train/val/test de uma vez
3. Escalar features
4. Salvar 3 arquivos `.pt` estáticos

Não há conceito de **streaming**, **janelas temporais**, ou **atualizações incrementais**.

### 2.4 Sem Mecanismo de Expiração

O modelo não distingue arestas recentes de arestas antigas. Em uma topologia real,
fluxos têm tempo de vida finito — um IP que não troca pacotes por horas deveria
ter suas arestas expiradas.

### 2.5 Sem Tratamento de Remoção de Nós

Não há suporte para remoção de nós/arestas. O grafo só cresce (se fosse dinâmico),
nunco encolhe.

---

## 3. Plano de Adaptação em Etapas

### Fase 1: Infraestrutura de Grafo Dinâmico

#### 1.1 Estrutura de Dados Temporal

**O quê:** Substituir o `Data` estático do PyG por um `TemporalData` ou estrutura
própria que mantenha múltiplos snapshots/intervalos de tempo.

**Como:**
- Criar uma classe `DynamicGraph` que armazena arestas com timestamps e permite
  consultas por janela temporal.
- Manter índices separados para: `edge_index` atual, `edge_attr`, timestamps,
  e um dicionário de nós ativos.

**Arquivos afetados:** `utils/dataloaders.py` (novo módulo: `utils/dynamic_graph.py`)

**Eficiência:** Adicionar/remover arestas deve ser O(1) amortizado. Usar
`defaultdict(list)` para arestas por nó e `set` para nós ativos.

**Tempo:** 2-3 dias de implementação.

#### 1.2 Mapeamento Dinâmico de Nós

**O quê:** Substituir o `node_map` fixo por um mapeamento que permite novos IPs
em tempo de execução.

**Como:**
- Manter um `dict` global IP→node_id que cresce sob demanda.
- Quando um IP não visto antes aparece, alocar um novo node_id e expandir
  as matrizes de embedding de nó.
- Usar `torch.nn.Embedding` com `padding_idx` para permitir crescimento.

**Arquivos afetados:** `utils/dataloaders.py`, `models/graphids.py`

**Eficiência:** Alocação O(1) para novos nós. Expansão de embedding requer
recriação da tabela (custo O(num_nós)) mas é rara.

**Segurança:** Prevenir exaustão de memória com número máximo de nós (DoS via
inundação de IPs falsos).

**Tempo:** 1-2 dias.

#### 1.3 Janelamento Temporal

**O quê:** Processar fluxos em janelas deslizantes (ex: 5 minutos) em vez de
tudo de uma vez.

**Como:**
- Modificar `NetFlowDataset` para aceitar um intervalo `[start_time, end_time]`.
- Cada janela produz um subgrafo com as arestas daquele período.
- As janelas podem ter sobreposição (ex: stride < window) para suavizar
  transições.

**Arquivos afetados:** `utils/dataloaders.py`

**Eficiência:** A sobreposição de janelas aumenta o processamento total em
O(N * window/stride). Idealmente stride = window para máxima eficiência.

**Tempo:** 2-3 dias.

---

### Fase 2: Adaptação do Modelo

#### 2.1 Inicialização de Novos Nós

**O quê:** Quando um novo nó aparece, sua representação inicial não pode ser
zero (perda de informação) nem aleatória (instabilidade).

**Como:**
- Inicializar o embedding do novo nó com a **média dos embeddings dos nós vizinhos**
  que já existiam.
- Alternativa: usar o embedding do nó mais similar (por feature de aresta).
- No `SAGELayer`, garantir que o `propagate()` lide corretamente com nós que
  têm zero arestas incidentes (retornar zero ou uma inicialização padrão).

**Arquivos afetados:** `models/graphids.py`

**Eficiência:** Cálculo O(grau_medio) para cada novo nó.

**Segurança:** Um atacante poderia tentar "envenenar" embeddings conectando
um nó malicioso a muitos nós legítimos. Mitigação: limitar taxa de conexão
por nó.

**Tempo:** 2-3 dias.

#### 2.2 Suporte a Remoção de Nós/Arestas

**O quê:** Implementar decaimento/expiração de arestas antigas e remoção de nós
inativos.

**Como:**
- Atribuir um **peso temporal** `w(t) = exp(-λ * idade)` para cada aresta.
- Modificar `SAGELayer.message()` para multiplicar `edge_attr` pelo peso temporal.
- A cada janela, remover arestas com idade > TTL máximo.
- Nós sem arestas por múltiplas janelas podem ser removidos (ou marcados
  como inativos).

**Arquivos afetados:** `models/graphids.py`, `utils/dynamic_graph.py`

**Eficiência:** A remoção em lote é O(num_arestas_expiradas). A ponderação
temporal adiciona uma multiplicação extra por aresta.

**Tempo:** 2-3 dias.

#### 2.3 Atualização de Embeddings sem Retreino Completo

**O quê:** Quando o grafo muda (novas arestas/nós), os embeddings mudam.
O modelo precisa se adaptar sem retreinar do zero.

**Como:**
- **Opção A (Incremental):** Executar _forward pass_ apenas nos nós afetados
  e suas vizinhanças (subgrafo). Requer modificação no SAGELayer para
  propagação local.
- **Opção B (Fine-tuning periódico):** Manter o modelo congelado entre janelas
  e fazer fine-tuning rápido a cada N janelas com os novos dados.
- **Opção C (Online Learning):** Usar SGD online — a cada novo batch de arestas,
  fazer um passo de otimização.

**Arquivos afetados:** `models/graphids.py`, `utils/trainers.py`

**Eficiência:** Opção A é a mais eficiente (O(vizinhos_afetados)), mas a mais
complexa. Opção B é a mais simples. Opção C é um meio-termo.

**Tempo:** 4-5 dias para a Opção A (recomendada).

---

### Fase 3: Pipeline de Inferência em Streaming

#### 3.1 Detecção em Tempo Real

**O quê:** Processar cada novo fluxo de rede assim que chega, atribuindo
score de anomalia.

**Como:**
- Manter o grafo atualizado em memória.
- Para cada novo fluxo (aresta):
  1. Adicionar ao grafo dinâmico.
  2. Executar encoder SAGE (pode ser apenas nos nós afetados).
  3. Executar Transformer AE na janela de embeddings mais recente.
  4. Calcular erro de reconstrução → score de anomalia.
  5. Se score > threshold, alertar.

**Arquivos afetados:** Novo módulo `inference/streaming.py`

**Eficiência:** A latência por fluxo deve ser < 1ms para aplicações em tempo
real. O Transformer AE é o gargalo (O(window_size²) devido à self-attention).

**Tempo:** 3-4 dias.

#### 3.2 Atualização Dinâmica do Threshold

**O quê:** O threshold de anomalia ajustado na validação pode não valer para
sempre, pois o comportamento da rede muda (conceito drift).

**Como:**
- Manter uma **janela deslizante de scores recentes** (ex: últimos 1000 fluxos).
- Recalcular o threshold periodicamente usando o método MAD (mediana + multiplier * MAD)
  sobre os scores da janela.
- Alternativa: usar detecção de drift (ex: Page-Hinkley) para acionar
  recalibração.

**Arquivos afetados:** `utils/trainers.py` (função `find_threshold`)

**Segurança:** Um atacante que injeta muitos fluxos anômalos lentamente pode
deslocar o threshold. Usar MAD com janela longa e limite mínimo de segurança.

**Tempo:** 1-2 dias.

---

### Fase 4: Treinamento Contínuo / Adaptação

#### 4.1 Estratégia de Retreinamento

**O quê:** Decidir quando e como retreinar o modelo com novos dados.

**Opções:**
- **Retreino periódico:** A cada N janelas, retreinar com dados acumulados.
  Simples, mas pode perder janelas de detecção crítica.
- **Retreino sob demanda:** Monitorar métricas de validação (PR-AUC em uma
  janela de referência). Quando cair abaixo de um limiar, acionar retreino.
- **Online Learning:** Atualizar o modelo a cada mini-batch de novas arestas.

**Tempo:** 3-4 dias para implementar retreino periódico + sob demanda.

#### 4.2 Prevenção de Catastrophic Forgetting

**O quê:** O modelo não pode esquecer padrões de ataque antigos ao aprender
novos.

**Como:**
- **Experience Replay:** Manter um buffer dos exemplos mais representativos
  de ataques passados e incluí-los no treinamento.
- **Elastic Weight Consolidation (EWC):** Regularizar pesos importantes para
  tarefas anteriores.
- **Ensemble:** Manter múltiplos modelos de diferentes períodos e votar.

**Arquivos afetados:** `utils/trainers.py`

**Eficiência:** Experience replay é o mais leve (custo de armazenamento linear).
EWC adiciona custo computacional por parâmetro.

**Tempo:** 3-4 dias.

---

### Fase 5: Tratamento de Casos Extremos

#### 5.1 Cold Start de Novos Nós

**Problema:** Um novo IP aparece, não tem arestas → embedding zero → detecção
ruim ou falsamente positiva.

**Solução:**
- Usar features contextuais (sub-rede, geolocalização, tipo de dispositivo)
  para bootstrap.
- Atribuir uma representação baseada na média de nós similares.
- Primeiras N arestas do novo nó são marcadas como "baixa confiança".

**Tempo:** 2-3 dias.

#### 5.2 Ataques de Graph Poisoning

**Problema:** Injetar arestas maliciosas para manipular embeddings e esconder
ataques.

**Solução:**
- Limitar grau máximo de nós novos.
- Detectar componentes desconexas suspeitas.
- Usar certifcados de robustez (certified robustness) para GNNs.

**Tempo:** 3-5 dias para implementação básica.

#### 5.3 Oscilação de Topologia (Flapping)

**Problema:** Nós entrando e saindo rapidamente em redes sem fio/IoT.

**Solução:**
- Implementar **histerese**: um nó só é removido após ficar inativo por
  M janelas consecutivas.
- Usar uma **representação temporal suave** (ex: média exponencial móvel
  do embedding) em vez de atualização instantânea.

**Tempo:** 1-2 dias.

---

## 4. Tabela Resumo das Implicações

| Componente | Eficiência | Tempo de Implementação | Segurança |
|---|---|---|---|
| Mapeamento dinâmico de nós | O(1) p/ novo nó | 1-2 dias | Limitar max nós (anti-DoS) |
| Janelamento temporal | O(N * window/stride) | 2-3 dias | Janela longa reduz detecção |
| Ponderação temporal | O(E) p/ forward | 1-2 dias | Ajuda contra ataques lentos |
| Inicialização de novos nós | O(grau_medio) | 2-3 dias | Risco de envenenamento |
| Propagação incremental | O(subgrafo_afetado) | 4-5 dias | Mesmo risco do modelo base |
| Detecção em streaming | <1ms/alvo | 3-4 dias | Evasão por injeção lenta |
| Threshold adaptativo | O(window_scores) | 1-2 dias | Deslocamento por ataques lentos |
| Retreinamento periódico | O(epochs * dados) | 3-4 dias | Esquece ataques raros |
| Cold start | O(1) | 2-3 dias | Falso positivo inicial |
| Graph Poisoning | O(grau) | 3-5 dias | **Crítico** — mitigação essencial |
| Oscilação (flapping) | O(1) | 1-2 dias | Estabilidade vs. detecção |

**Estimativa total:** 25-40 dias de desenvolvimento para uma implementação
completa e robusta.

---

## 5. Arquitetura Proposta (Visão Geral)

```
[Fluxo de Rede] ──► [Fila de Mensagens (Kafka/RabbitMQ)]
                           │
                           ▼
              ┌─────────────────────────┐
              │  StreamingGraphBuilder  │
              │  - Adiciona arestas     │
              │  - Expira arestas       │
              │  - Gerencia nós         │
              │  - Janelas temporais    │
              └──────────┬──────────────┘
                         │
                         ▼
              ┌─────────────────────────┐
              │   GNN Encoder (SAGE)    │◄── Atualização incremental
              │   - Embedding de nós    │     só em nós afetados
              │   - Embedding de arestas│
              └──────────┬──────────────┘
                         │
                         ▼
              ┌─────────────────────────┐
              │ Transformer Autoencoder │
              │   - Janela deslizante   │
              │   - Erro reconstrução   │
              └──────────┬──────────────┘
                         │
                         ▼
              ┌─────────────────────────┐
              │   Anomaly Scorer        │
              │   - Threshold dinâmico  │
              │   - Alerta se > limiar  │
              └──────────┬──────────────┘
                         │
                         ▼
                   [Alerta/Log]
```

---

## 6. Recomendações Finais

### Prioridade de Implementação

1. **Fase 1 (Infraestrutura dinâmica)** — base para todo o resto.
2. **Fase 3 (Inferência em streaming)** — maior impacto prático imediato.
3. **Fase 2.2 (Remoção/expiração)** — essencial para topologia real.
4. **Fase 4 (Treinamento contínuo)** — necessário para manter acurácia.
5. **Fase 5 (Casos extremos)** — robustez contra ataques e cenários
   adversos.

### Trade-offs Principais

- **Latência vs. Acurácia:** Janelas maiores melhoram a detecção mas
  aumentam latência. Recomenda-se janelas de 10-60 segundos para redes
  típicas.
- **Memória vs. Completeza:** Manter histórico completo de arestas é
  caro. Usar TTL (time-to-live) de 1 hora para arestas, com resampling
  para arestas antigas importantes.
- **Atualização vs. Estabilidade:** Atualizar embeddings a cada nova
  aresta causa oscilação. Recomenda-se atualização em mini-batches a
  cada 100-1000 novas arestas ou a cada 1 segundo, o que ocorrer primeiro.
- **Segurança vs. Usabilidade:** Medidas contra graph poisoning (limitar
  grau, certificados de robustez) podem aumentar falsos positivos.
  Ajuste fino é necessário.
