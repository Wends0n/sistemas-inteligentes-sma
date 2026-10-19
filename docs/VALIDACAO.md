# Validação e rastreabilidade

Ambiente: Linux, GCC 13.3.0; NumPy 2.3.5; Matplotlib 3.10.8. Data da entrega:
19/10/2026. Fontes externas conferidas: demo dos autores SMA e OR-Library.

## Comandos reproduzíveis

```bash
make test
python3 scripts/check_reference.py
make sanitize
make experiment
make plot
```

Em ambientes restritos onde LeakSanitizer não consegue acessar `/proc`, como
ocorreu nesta execução, foi usado:

```bash
ASAN_OPTIONS=detect_leaks=0 make sanitize
```

AddressSanitizer e UndefinedBehaviorSanitizer passaram sem diagnóstico. A
verificação de vazamentos do LeakSanitizer ficou desativada por limitação do
ambiente; não se declara cobertura de vazamentos por esse teste.

Saída dos testes comuns:

```text
Exaustao: 720 permutacoes conferidas; otimo pequeno=353, SMA=353
OK: RNG, Taillard, SPV, makespan, pesos, limites, empate, contagem, reproducibilidade e parametros.
OK: 36 pontos de histórico C conferem com oráculo independente das equações.
OK: 60 soluções, 30000 pontos e estatísticas validados.
```

## Conferência independente dos dados Taillard

O arquivo `flowshop2.txt` da OR-Library contém procedimentos, sementes e um
programa C completo. Foi extraído o programa, selecionado `ANSI_C=1`, compilado
e executado fora do solver. Suas linhas são tarefas com pares máquina/tempo.
As colunas de tempos foram transpostas e comparadas às matrizes exportadas:

```text
ta001: todos os 100 tempos conferem com gerador C da OR-Library
ta011: todos os 200 tempos conferem com gerador C da OR-Library
```

O programa de terceiros não participa da construção do solver e não está
incluído como código próprio. As matrizes resultantes estão em `data/` e
`results/`. A matriz de verificação ta001 também consta explicitamente em
`tests/test.c`, conforme a fonte oficial.

## O que cada verificação estabelece

| Verificação | Evidência |
|---|---|
| Matriz oficial e sementes | Identidade dos benchmarks usados |
| Exaustão 6! + avaliação independente | Recorrência do makespan correta nos casos verificados |
| SPV com empates e permutações finais | Soluções completas sem repetição de tarefas |
| 36 pontos contra oráculo | Tradução consistente das equações e ordem dos sorteios testadas |
| Mesma semente, mesmo histórico | Reprodutibilidade na mesma configuração |
| N*T chamadas | Orçamento declarado efetivamente aplicado |
| 60 reavaliações independentes | Fitness exportado corresponde à solução exportada |
| 30.000 registros | Integridade e monotonicidade do melhor histórico |
| Sanitizadores disponíveis | Ausência de falhas detectadas de memória/UB nos testes executados |

O oráculo Python usa o mesmo gerador pseudoaleatório e não é uma execução do
MATLAB. A validação não equivale a uma prova formal nem a equivalência bit a bit
com todos os resultados possíveis da demo original.

## Somas de verificação

`SHA256SUMS.txt`, na raiz, registra os arquivos entregues (exceto ele mesmo e
executáveis temporários). Para verificar os bytes originais no Linux:

```bash
sha256sum -c SHA256SUMS.txt
```

Executar novamente o experimento modifica pelo menos os tempos de CPU em CSV e
pode modificar metadados dos gráficos. Verifique os hashes antes de reproduzir.
