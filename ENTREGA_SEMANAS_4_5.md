# Entrega das Semanas 4 e 5: o que foi feito e onde estão os resultados

Projeto: **Slime Mould Algorithm (SMA) em C aplicado ao Flowshop Permutacional**
(instâncias Taillard `ta001` e `ta011`). O SMA base e a sintonia com Optuna
(Semana 3) vêm de entregas anteriores. Esta entrega acrescenta:

- **Semana 4 (30/10):** Surrogate MLP no lugar da função objetivo.
- **Semana 5 (06/11):** controle fuzzy adaptativo do parâmetro `z` do solver.

Detalhes completos: [`docs/SEMANAS_4_5.md`](docs/SEMANAS_4_5.md).
Para reproduzir tudo: `pip install -r requirements-advanced.txt` e `make semanas45`.

---

## Semana 4: Treino do Surrogate (Rede Neural para a Função Objetivo)

### O que foi feito
1. **Amostragem LHS:** 1500 vetores por instância (dimensão 20, domínio `[-1, 1]`)
   gerados por Latin Hypercube Sampling. Cada vetor é decodificado por SPV e
   avaliado com o makespan real.
2. **Treino da RNA:** MLP `20 → 32 → 16 → 1` (ReLU nas ocultas, saída linear,
   Adam, early stopping), um modelo por instância, divisão 80/20 treino/teste.
3. **Integração ao solver:** os pesos são exportados para C
   (`src/surrogate_model.c`) e o SMA passa a usar `surrogate_objective` no loop,
   sem Python. A melhor solução encontrada é **reavaliada com a função real**.
4. **Comparativo de tempo:** avaliação real vs surrogate, com o objetivo nativo
   e com um objetivo "caro" emulado (repete o cálculo do makespan R vezes, para
   simular o custo de CFD/FEA).

### Resultados principais
Métricas de regressão no conjunto de teste (300 amostras por instância):

| Instância | MSE | RMSE | R² |
|---|---:|---:|---:|
| ta001 | 2382,4 | 48,8 | 0,359 |
| ta011 | 5276,7 | 72,6 | 0,239 |
| Média | 3829,5 | 60,7 | 0,299 |

- O R² é baixo porque o makespan depende de `argsort`, função descontínua que
  uma MLP suave aprende mal com poucos dados. A curva de aprendizado mostra
  melhora lenta com mais amostras.
- Consequência: as soluções do surrogate, reavaliadas no objetivo real, são
  piores (ta001: 1400,0 vs 1307,8; ta011: 1882,6 vs 1685,8).
- **Ganho de tempo:** com o objetivo nativo (~1 µs por avaliação) o surrogate
  **não dá ganho** (ficou 22–26% mais lento). Com objetivo caro emulado o ganho
  cresce com o custo: de 17× a 1202× (100× a 5000× de custo), e a partir de
  ~1000× o custo de construir o surrogate se paga em uma única execução.
- O R² de 0,945 de uma versão anterior (duas instâncias empilhadas) **não deve
  ser usado**: é inflado pela diferença de médias entre as instâncias.

### Onde estão
| Item | Arquivo |
|---|---|
| Pipeline LHS + treino da RNA | `scripts/surrogate_fuzzy.py` |
| Dataset amostrado | `results/surrogate/dataset_lhs.csv` |
| Métricas MSE, RMSE, R² | `results/surrogate/metricas_rna.json` |
| Gráfico real vs predito | `results/surrogate/regressao_real_vs_predito.png` |
| Curva de aprendizado | `results/surrogate/curva_aprendizado.csv` |
| Tempo real vs surrogate (objetivo nativo) | `results/surrogate/comparacao_tempo.csv` |
| Tempo real vs surrogate (objetivo caro) | `results/surrogate/comparacao_tempo_custo.csv` |
| Descrição do modelo | `results/surrogate/modelo_info.json` |
| Inferência da MLP em C | `src/surrogate_model.c` |

---

## Semana 5: Controle Fuzzy Adaptativo do Solver

### O que foi feito
1. **Controlador Takagi-Sugeno de ordem zero** que ajusta o parâmetro `z`
   (probabilidade de reinicialização) **a cada geração**.
2. **Entradas** (variáveis de estado do solver):
   - Diversidade `D`: desvio-padrão médio da população, normalizado pelo domínio.
   - Estagnação `E`: gerações consecutivas sem melhora, normalizadas.
3. **Funções de pertinência:** rampas lineares, "baixa" e "alta" para cada entrada.
4. **Base de regras** (4 regras, saída `z = Σ(wᵢ·zᵢ)/Σ(wᵢ)`):

| Regra | Diversidade | Estagnação | z |
|---|---|---|---:|
| R1 | Baixa | Baixa | 0,0325 |
| R2 | Baixa | Alta | 0,0681 |
| R3 | Alta | Baixa | 0,0120 |
| R4 | Alta | Alta | 0,0619 |

5. **Uso dos dados do Optuna:** os valores de saída (singletons de `z`) são
   quantis dos 25% melhores trials do Optuna; os termos de entrada vêm de
   percentis de 20 execuções do solver (sementes de calibração diferentes das
   de avaliação).
6. **Solver integrado:** a cada geração calcula `D`, atualiza `E`, calcula `z`
   pelo controlador, aplica na reinicialização e registra os valores.

### Resultados principais
- `z` varia de 0,012 a 0,068 ao longo das gerações: fica baixo no início
  (refinamento) e sobe quando a estagnação cresce.
- Fuzzy vs `z` estático (objetivo real, 30 execuções):

| Instância | Configuração | Makespan médio | Melhor |
|---|---|---:|---:|
| ta001 | Estático z=0,03 | 1297,7 | 1283 |
| ta001 | Estático z=0,0164 (Optuna) | 1298,6 | 1286 |
| ta001 | **Fuzzy** | 1300,7 | 1286 |
| ta011 | Estático z=0,03 | 1666,0 | 1619 |
| ta011 | Estático z=0,0164 (Optuna) | 1653,3 | 1618 |
| ta011 | **Fuzzy** | 1655,5 | 1615 |

- Os testes de Wilcoxon não mostram diferença significativa a 5%. O fuzzy é
  competitivo e interpretável, mas não supera de forma comprovada a sintonia
  estática nestas instâncias pequenas (a superfície do Optuna em `z` é quase plana).

### Onde estão
| Item | Arquivo |
|---|---|
| Calibração com Optuna, comparação e gráficos | `scripts/fuzzy_optuna.py` |
| Parâmetros calibrados | `results/fuzzy/calibracao_optuna.json` |
| Funções de pertinência (figura) | `results/fuzzy/funcoes_pertinencia.png` |
| Variáveis linguísticas e base de regras | `docs/SEMANAS_4_5.md` |
| Solver com fuzzy a cada geração | `src/sma.c` (`sma_solve_ex`, `sma_solve_adaptive`) e `include/sma.h` |
| **Gráfico temporal de `z` por geração** | `results/fuzzy/variacao_z_geracoes.png` |
| Dados do gráfico (z, D, E por geração) | `results/fuzzy/historico_controlador.csv` |
| Fuzzy com surrogate no loop | `results/fuzzy/variacao_z_surrogate.png` e `historico_controlador_surrogate.csv` |
| Fuzzy vs estático | `results/fuzzy/comparacao_fuzzy_estatico.csv` |
| Testes estatísticos pareados | `results/fuzzy/testes_pareados_fuzzy.csv` |
| Convergência média | `results/fuzzy/comparacao_convergencia_fuzzy.png` |
| Resumo | `results/fuzzy/resumo_fuzzy.json` e `results/semanas_4_5_resumo.json` |

---

## Arquivos de semanas anteriores (base do projeto)
- SMA em C e Flowshop: `src/sma.c`, `src/pfs.c`, `src/rng.c`, `README.md`.
- Resultados do SMA base: `results/runs.csv`, `summary.csv`, `convergencia.png`.
- Sintonia com Optuna (Semana 3): `scripts/optuna_tune.py`, `README_OPTUNA.md`, `results/optuna/`.
