# Otimização de hiperparâmetros do SMA-PFS com Optuna

## 1. Objetivo

Este trabalho integra o framework **Optuna** ao solver **Slime Mould Algorithm (SMA)** aplicado ao problema de **Permutation Flowshop Scheduling (PFS)**. O solver continua implementado em C; o script Python coordena a compilação, a execução dos experimentos, a busca dos hiperparâmetros e a geração dos resultados.

O objetivo da otimização é encontrar uma configuração fixa de hiperparâmetros que **minimize o makespan final médio** nas instâncias de teste utilizadas. Makespan é o tempo total necessário para concluir o conjunto de tarefas no escalonamento; valores menores representam soluções melhores.

## 2. Estrutura de pastas

Para apenas entregar o script e os resultados, a pasta pode ser organizada assim:

```text
entrega-optuna/
├── scripts/
│   └── optuna_tune.py
├── resultados/
│   ├── importancia_hiperparametros.png
│   ├── historico_otimizacao.png
│   ├── comparacao_baseline_tunada.csv
│   ├── resultado_optuna.json       # registro complementar
│   └── trials.csv                  # registro complementar de cada tentativa
└── README.md
```

Para conseguir rodar o script adequadamente, é recomendado que os diretorios sejam mesclados aos da entrega da primeira parte, a versao com todo o projeto, inclusive optuna implementado, esta no github: https://github.com/Wends0n/sistemas-inteligentes-sma

**Observação:** essa estrutura serve para avaliação dos entregáveis. Para executar o script de forma independente, também são necessários o solver C e seus arquivos relacionados, o `Makefile`, as instâncias de teste e as dependências Python. Nesse caso, mantenha a estrutura original do projeto, incluindo `src/`, `include/`, `data/` e `Makefile`; o script grava os resultados em `results/optuna/` por padrão.

## 3. Hiperparâmetros otimizados

O script otimiza dois parâmetros que já fazem parte da implementação do SMA:

- **`population` — tamanho da população:** número de soluções candidatas mantidas pelo algoritmo em cada iteração. O espaço de busca vai de **10 a 60**, em incrementos de **5**. Uma população maior pode explorar mais soluções, mas também aumenta o custo computacional de cada iteração.
- **`z` — probabilidade de reinicialização:** probabilidade usada pelo SMA para reinicializar/alterar uma solução conforme a implementação do solver. O espaço de busca é contínuo entre **0,00 e 0,10**. Valores diferentes podem alterar o equilíbrio entre manter soluções existentes e introduzir variação na busca.

O Optuna utiliza o sampler **TPE (Tree-structured Parzen Estimator)** para propor configurações com base nos resultados das tentativas anteriores. A configuração encontrada é fixa: o algoritmo não muda esses hiperparâmetros durante uma execução do solver.

Parâmetros como `F`, `c1` e `c2` não foram incluídos porque são associados a outros algoritmos, como Differential Evolution e Particle Swarm Optimization, e não correspondem aos parâmetros utilizados nas equações desta implementação do SMA.

## 4. Função objetivo

Cada configuração é avaliada nas instâncias **Taillard `ta001` e `ta011`**. Para cada instância, o solver é executado várias vezes e calcula-se a média do melhor makespan final obtido. A função objetivo é a média simples das médias das duas instâncias:

```text
objetivo = (média_makespan_ta001 + média_makespan_ta011) / 2
```

O Optuna trabalha em modo de **minimização**. Portanto, entre duas configurações, prefere-se aquela com menor valor da função objetivo. Como a função usa os makespans absolutos das duas instâncias, ela não é uma métrica normalizada entre instâncias de escalas diferentes.

## 5. Configuração padrão (Baseline) e configuração tunada

A configuração **Baseline** usa os valores definidos como referência antes da otimização:

- `population = 30`
- `z = 0.03`

A configuração **Tunada** utiliza os valores de `population` e `z` do trial com o menor valor da função objetivo durante o estudo. O arquivo `comparacao_baseline_tunada.csv` apresenta os parâmetros e os makespans médios de cada instância, além do objetivo médio.

Para que a comparação seja justa, Baseline e Tunada são avaliadas com o mesmo número de execuções e iterações por execução dentro de uma rodada do script.

## 6. Como os resultados foram obtidos

O script `scripts/optuna_tune.py` realiza as seguintes etapas:

1. Compila o solver em C por meio de `make all`.
2. Avalia a configuração Baseline.
3. Cria um estudo Optuna com direção `minimize` e sampler TPE.
4. Em cada trial, seleciona `population` e `z`, executa o solver nas duas instâncias e calcula a função objetivo.
5. Reavalia a melhor configuração encontrada e compara seus resultados com a Baseline.
6. Salva os gráficos, a tabela comparativa e os arquivos complementares em `results/optuna/`.

Os arquivos principais são:

- **`historico_otimizacao.png`:** mostra o valor da função objetivo de cada trial e o melhor valor acumulado até aquele ponto.
- **`importancia_hiperparametros.png`:** mostra a importância relativa estimada pelo Optuna para os hiperparâmetros avaliados.
- **`comparacao_baseline_tunada.csv`:** tabela com a configuração padrão e a configuração encontrada pelo Optuna.
- **`resultado_optuna.json`:** resumo dos parâmetros encontrados e das condições usadas no experimento.
- **`trials.csv`:** registro dos trials executados e seus resultados.

A quantidade de trials, o número de execuções por instância e o limite de iterações podem ser definidos pela linha de comando. Os resultados incluídos devem ser interpretados de acordo com esses valores: uma busca curta demonstra a integração, mas não garante que os parâmetros encontrados sejam globalmente ótimos ou estatisticamente superiores.

## 7. Requisitos para executar o experimento

- Python 3.
- Compilador C compatível com C11 e `make`.
- Bibliotecas Python: `optuna`, `scikit-learn`, `matplotlib` e as dependências listadas nos arquivos `requirements.txt` e `requirements-optuna.txt` do projeto.
- Código-fonte do solver, cabeçalhos, `Makefile` e instâncias `ta001` e `ta011`.

Na raiz do projeto completo, a instalação e execução podem ser feitas assim:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-optuna.txt
python scripts/optuna_tune.py --trials 30 --runs 5 --iterations 300
```

Parâmetros da execução:

- `--trials`: quantidade de configurações testadas pelo Optuna.
- `--runs`: número de execuções do solver por instância para avaliar cada configuração.
- `--iterations`: limite de iterações de cada execução do solver.
- `--seed`: semente do sampler Optuna para tornar a sequência de sugestões reproduzível.

Os valores acima são um exemplo de execução mais ampla. Para reproduzir exatamente os arquivos de resultados entregues, consulte os parâmetros registrados em `resultado_optuna.json`.

## 8. Limitações e interpretação

- A importância dos hiperparâmetros é uma estimativa baseada nos trials realizados e pode variar, especialmente com poucas tentativas.
- Uma configuração selecionada e avaliada com as mesmas sementes pode se ajustar aos cenários observados. Para uma avaliação mais rigorosa, recomenda-se testar a configuração final com sementes independentes.
- A população altera o número de soluções avaliadas por iteração. Como o script mantém fixo o limite de iterações, configurações com populações maiores podem consumir mais avaliações computacionais.
- A configuração tunada é a melhor encontrada dentro do espaço de busca e do orçamento experimental utilizado; isso não prova que seja o ótimo global.
