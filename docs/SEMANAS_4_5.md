# Semanas 4 e 5 — Surrogate MLP e controle fuzzy adaptativo

## Escopo e integração

Esta extensão preserva o problema de Flowshop Permutacional (PFS), o SMA em C11,
a representação por chaves reais e o decodificador SPV. A função objetivo real
continua sendo o makespan calculado por `pfs_objective`.

- **Semana 4:** modelo substituto MLP treinado com amostras LHS, exportado como
  pesos em `src/surrogate_model.c`. A inferência acontece em C dentro do callback
  do SMA (sem Python no loop). Há um modelo por instância (`ta001`, `ta011`).
- **Semana 5:** controlador fuzzy Takagi-Sugeno que ajusta `z` a cada geração
  (`sma_solve_ex` / `sma_solve_adaptive`). Os parâmetros do controlador são
  **calibrados a partir dos trials do Optuna** e de execuções do próprio solver.

Scripts: `scripts/smalib.py` (utilidades ctypes/PFS), `scripts/surrogate_fuzzy.py`
(Semana 4 e integração fuzzy+surrogate) e `scripts/fuzzy_optuna.py` (Semana 5).

## Semana 4 — Surrogate

### Pipeline
1. Para cada instância, `--samples` vetores de dimensão 20 por
   `scipy.stats.qmc.LatinHypercube`, escalados para `[-1, 1]` (domínio do SMA).
2. Cada vetor é decodificado por SPV e avaliado com o makespan real; o dataset
   guarda as 20 coordenadas e `makespan_real`.
3. Divisão 80/20 treino/teste por instância; `StandardScaler` em X e y.
4. MLP `20 → 32 → 16 → 1` (ReLU nas ocultas, saída linear, Adam, early stopping).
5. MSE, RMSE e R² **por instância**, no conjunto de teste.
6. Pesos exportados para C; o SMA recebe `surrogate_objective` no lugar de
   `pfs_objective`. A solução do surrogate é **reavaliada pela função real**.

### Métricas de regressão (teste, 300 amostras por instância)

| Instância | MSE | RMSE | R² |
|---|---:|---:|---:|
| ta001 | 2382,4 | 48,8 | 0,359 |
| ta011 | 5276,7 | 72,6 | 0,239 |
| **Média** | 3829,5 | 60,7 | **0,299** |

> **Correção importante.** A versão anterior reportava R² = 0,945 calculado sobre
> as predições das duas instâncias empilhadas. Esse valor é inflado pela
> diferença entre as médias de ta001 (~1500) e ta011 (~2000) e não mede a
> qualidade do surrogate; ele permanece em `metricas_rna.json` apenas marcado
> como "não usar". A métrica válida é a por instância.

**Interpretação.** O surrogate explica só ~24–36% da variância do makespan. O
motivo é estrutural: o makespan depende de `argsort(x)`, função descontínua nas
chaves reais, e uma MLP suave aprende isso mal com poucos dados. A curva de
aprendizado (`results/surrogate/curva_aprendizado.csv`, gerada com
`--learning-curve`) mostra melhora lenta com mais amostras (R² de ta001 de 0,40
com 500 para 0,49 com 6000; ta011 de −0,03 para 0,31), isto é, mais dados ajudam,
mas o ganho é modesto. Consequência prática, já visível nos experimentos: as
soluções achadas via surrogate e reavaliadas no objetivo real são piores
(ta001: 1400,0 vs 1307,8; ta011: 1882,6 vs 1685,8). Arquiteturas mais
adequadas (por exemplo, features baseadas em ranking) são trabalho futuro.

### Comparativo de desempenho e ganho computacional

População 20, 100 gerações (2000 avaliações), medianas de 5 execuções
(`comparacao_tempo.csv`).

**Objetivo nativo (makespan 20×5/20×10, ~1 µs/avaliação):** o surrogate **não dá
ganho**: nesta medição ele ficou ~22–26% *mais lento* (1,40 vs 1,15 µs/aval. em
ta001; 1,50 vs 1,19 em ta011), pois a inferência da rede custa tanto quanto o
próprio makespan. Medições anteriores, com 4–9% de "ganho", estavam dentro do
ruído de medição. Esse é o comportamento esperado quando o objetivo é barato.

**Objetivo caro simulado:** para mostrar o cenário que motiva o surrogate (CFD/FEA),
`pfs_objective_costly` repete o cálculo do makespan `R` vezes (mesmo valor,
custo maior). É uma **emulação** do custo, não uma simulação física
(`comparacao_tempo_custo.csv`; "construção" = coletar o dataset LHS com o
objetivo caro + treinar a rede):

| Instância | Custo por avaliação | Tempo real (s) | Tempo surrogate (s) | Speedup | Construção (s) | Execuções p/ empatar |
|---|---:|---:|---:|---:|---:|---:|
| ta001 | 100× (0,024 ms) | 0,048 | 0,0028 | 17,1× | 0,28 | 6,3 |
| ta001 | 1000× (0,24 ms) | 0,481 | 0,0028 | 171× | 0,61 | 1,3 |
| ta001 | 5000× (1,19 ms) | 2,37 | 0,0028 | 846× | 2,03 | 0,86 |
| ta011 | 100× (0,033 ms) | 0,065 | 0,0030 | 21,7× | 0,24 | 3,9 |
| ta011 | 1000× (0,30 ms) | 0,606 | 0,0030 | 202× | 0,65 | 1,1 |
| ta011 | 5000× (1,80 ms) | 3,60 | 0,0030 | 1202× | 2,89 | 0,80 |

Leitura: o ganho cresce linearmente com o custo do objetivo real; já a partir de
~1000× (≈0,3 ms/avaliação) o custo de construir o surrogate se paga em uma única execução. Esses
tempos valem para o hardware em que foram medidos e variam entre execuções;
o ganho de tempo vem sempre acompanhado da perda de qualidade descrita acima, e
a predição não deve ser tratada como validação final.

## Semana 5 — Controlador fuzzy adaptativo

### Calibração com os dados do Optuna
`scripts/fuzzy_optuna.py` calcula todos os parâmetros a partir de dados
(`results/fuzzy/calibracao_optuna.json`):

1. **Saídas (singletons de z):** quantis de `z` dos 25% melhores trials
   (8 de 30) de `results/optuna/trials.csv`: q10 = 0,0120, q50 = 0,0325,
   q75 = 0,0619, q90 = 0,0681. Atribuição: R3 = q10 (refinar), R1 = q50,
   R4 = q75, R2 = q90 (explorar).
2. **Termos de entrada:** percentis das variáveis de estado observadas em 20
   execuções estáticas do solver (sementes 100–109, objetivo real).
   "Baixa" vai de P10 a P60 (decrescente) e "alta" de P40 a P90 (crescente).
3. **Janela de estagnação:** percentil 80 das gerações sem melhora observadas
   (147 de 300), resultando em 0,49·T.

As sementes de calibração (100–109) são diferentes das de avaliação (5000+).

> **Por que a calibração mudou.** O controlador anterior usava limiares
> escolhidos à mão. Com a diversidade real do SMA (mediana ≈ 0,12) a diversidade
> era sempre "baixa", e a estagnação saturava em 1 após 30 gerações; por isso `z`
> ficava em 0,045 (singleton de R1) em ~58% das gerações. Agora R1 responde por
> 8–20% das gerações.

### Variáveis linguísticas e funções de pertinência
Entradas (recalculadas a cada geração):
- **Diversidade D:** média, nas 20 dimensões, do desvio-padrão da população
  dividido pela amplitude do domínio, em `[0,1]`.
- **Estagnação E:** gerações consecutivas sem melhora estrita /
  `max(10, 0,49·T)`, em `[0,1]`.

Termos (rampas lineares, figura em `results/fuzzy/funcoes_pertinencia.png`):

| Termo | Forma | Parâmetros |
|---|---|---|
| D baixa | decrescente | 1 em D ≤ 0,056; 0 em D ≥ 0,132 |
| D alta | crescente | 0 em D ≤ 0,108; 1 em D ≥ 0,198 |
| E baixa | decrescente | 1 em E ≤ 0,014; 0 em E ≥ 0,537 |
| E alta | crescente | 0 em E ≤ 0,204; 1 em E ≥ 1,000 |

### Base de regras (Takagi-Sugeno de ordem zero)

| Regra | Diversidade | Estagnação | z (singleton) |
|---|---|---|---:|
| R1 | Baixa | Baixa | 0,0325 |
| R2 | Baixa | Alta | 0,0681 |
| R3 | Alta | Baixa | 0,0120 |
| R4 | Alta | Alta | 0,0619 |

Disparo = produto das pertinências; saída `z = Σ(wᵢ·zᵢ)/Σ(wᵢ)` (0,04 se Σwᵢ ≈ 0).
Lógica: diversidade baixa e/ou estagnação alta pedem reinicialização
(exploração); diversidade alta e estagnação baixa pedem refinamento.

### Integração no solver
`include/sma.h`: `FuzzyParams`, `sma_fuzzy_defaults`, `sma_fuzzy_z` e
`sma_solve_ex` (com `fp == NULL` roda com z fixo e ainda registra z, D e E por
geração). `sma_solve_adaptive` continua existindo e usa os parâmetros padrão
(que coincidem com a calibração; o script verifica isso). A cada geração o
solver: (1) calcula D, (2) atualiza E, (3) calcula `z` pelo controlador,
(4) aplica `z` à reinicialização, (5) registra os valores.
`sma_solve` (z estático) permanece intacto.

### Resultado temporal
`results/fuzzy/variacao_z_geracoes.png` mostra, para ta001 e ta011 (objetivo
real, 30 agentes, 300 gerações), `z` (com a linha do z estático 0,03), D e E por
geração. `z` varia de 0,012 a 0,068 (~100 valores distintos por instância):
fica baixo nas fases iniciais (refinamento) e sobe quando a estagnação cresce.
Os dados estão em `historico_controlador.csv`. A execução com o surrogate no
loop está em `historico_controlador_surrogate.csv` e `variacao_z_surrogate.png`.

### Fuzzy vs. z estático (objetivo real, 30 execuções, mesma população e sementes)

| Instância | Configuração | Makespan médio | Desvio | Melhor |
|---|---|---:|---:|---:|
| ta001 | Estático z=0,03 | 1297,7 | 4,1 | 1283 |
| ta001 | Estático z=0,0164 (Optuna) | 1298,6 | 7,4 | 1286 |
| ta001 | **Fuzzy** | 1300,7 | 9,1 | 1286 |
| ta011 | Estático z=0,03 | 1666,0 | 30,6 | 1619 |
| ta011 | Estático z=0,0164 (Optuna) | 1653,3 | 20,0 | 1618 |
| ta011 | **Fuzzy** | 1655,5 | 27,8 | 1615 |

Testes pareados de Wilcoxon (`testes_pareados_fuzzy.csv`): nenhuma diferença é
estatisticamente significativa a 5% (ta001: p = 0,13 e 0,23; ta011: p = 0,07
contra o baseline, em que o fuzzy tem média 10,5 menor, e p = 0,52 contra o z do
Optuna). **Conclusão honesta:** o fuzzy é competitivo e adapta `z` de forma
interpretável, mas não supera de modo comprovado a sintonia estática nestas
instâncias pequenas; o próprio estudo do Optuna mostra uma superfície quase
plana em `z` (objetivos entre 1466 e 1484), o que limita o ganho possível.
A convergência média está em `comparacao_convergencia_fuzzy.png`.

## Como reproduzir

```bash
python3 -m pip install -r requirements-advanced.txt
make clean all test                       # testes em C (inclui o fuzzy)
python3 scripts/fuzzy_optuna.py           # Semana 5: calibração, comparação, gráficos
python3 scripts/surrogate_fuzzy.py --learning-curve   # Semana 4 (+ curva de aprendizado)
```
(ou `make semanas45`). Rode `fuzzy_optuna.py` antes de `surrogate_fuzzy.py` para
que `semanas_4_5_resumo.json` inclua o resumo do fuzzy. Parâmetros padrão:
`surrogate_fuzzy.py --samples 1500 --population 20 --iterations 100 --seed 20261114`;
`fuzzy_optuna.py --population 30 --iterations 300 --calib-seeds 10 --eval-seeds 30`.

Arquivos gerados: `results/surrogate/{dataset_lhs.csv, metricas_rna.json,
modelo_info.json, comparacao_tempo.csv, comparacao_tempo_custo.csv,
curva_aprendizado.csv, regressao_real_vs_predito.png}`, `src/surrogate_model.c`,
`results/fuzzy/{calibracao_optuna.json, funcoes_pertinencia.png,
variacao_z_geracoes.png, historico_controlador.csv,
historico_controlador_surrogate.csv, variacao_z_surrogate.png,
comparacao_fuzzy_estatico.csv, testes_pareados_fuzzy.csv,
comparacao_convergencia_fuzzy.png, resumo_fuzzy.json}` e
`results/semanas_4_5_resumo.json`.
