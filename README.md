# SMA em C aplicado ao Flowshop Permutacional

Implementação do Slime Mould Algorithm (SMA) do zero em **C11**, modularizada,
com avaliação experimental em **ta001 e ta011**, 30 execuções por instância.
O solver usa somente a biblioteca padrão de C e `libm`, sem frameworks de otimização.

## O que está sendo implementado

O SMA original é contínuo; o PFS exige uma permutação. Portanto, a descrição
precisa desta implementação é **SMA contínuo com decodificação por chaves reais
(SPV)**. Os operadores do núcleo seguem a demo SMA.m versão 1.0 e as convenções
numéricas do documento fornecido. O decodificador é a adaptação para o PFS,
não um operador do artigo original nem uma alegada versão discreta canônica.
Não há busca local, NEH na população, cruzamento ou mutação adicionados ao SMA.

O documento propõe Sphere, Rastrigin e Ackley; a solicitação atual pede PFS.
Assim, os dois testes principais são duas **instâncias** de Taillard, isto é,
duas funções objetivo `f_ta001(x)` e `f_ta011(x)` que retornam o makespan da
permutação decodificada. Sphere aparece apenas em testes de verificação do núcleo.

## Início rápido no Linux Mint

Pré-requisitos: compilador C11 (GCC/Clang), Make; Python com NumPy e Matplotlib
somente para análise e desenho. O executável do solver não depende de Python.

Dentro da pasta extraída:

```bash
make
make test
python3 scripts/check_reference.py
make experiment
make plot
```

`make experiment` executa **30 × 500 × 30 avaliações em cada instância** e
substitui os resultados da pasta `results`. São 900.000 avaliações no total.
Os resultados já estão incluídos no pacote; não é necessário executar novamente
para visualizar `results/convergencia.png` ou `results/convergencia.svg`.

Caso precise instalar as dependências de gráficos em um ambiente virtual:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Execução alternativa, com pasta já existente:

```bash
mkdir -p novos-resultados
./build/sma_pfs novos-resultados 30 500 30
python3 scripts/analyze.py novos-resultados
```

Argumentos: pasta, quantidade de execuções, iterações, população. Para reproduzir
o experimento entregue, use exatamente os valores acima. Instâncias, domínio,
`z` e política de sementes estão documentados em `src/main.c`.

No Windows, pode-se compilar com GCC/MinGW no terminal, a partir da raiz:

```powershell
gcc -O2 -std=c11 -Wall -Wextra -Wpedantic -Iinclude src/main.c src/sma.c src/pfs.c src/rng.c -lm -o sma_pfs.exe
.\sma_pfs.exe results 30 500 30
python scripts/analyze.py results
```

## Organização do código

| Arquivo | Responsabilidade |
|---|---|
| `include/sma.h`, `src/sma.c` | Contrato genérico de minimização e núcleo SMA: fronteiras, fitness, ranking, pesos, melhor histórico e movimentos. |
| `include/rng.h`, `src/rng.c` | SplitMix64, uniformes de 53 bits e seleção de índices sem viés de módulo. |
| `include/pfs.h`, `src/pfs.c` | Geração Taillard, SPV, validação de permutação e makespan. |
| `src/main.c` | Executa cada repetição com estado novo e exporta CSVs e instâncias. |
| `tests/test.c` | Testes unitários, exaustão de instância pequena e traços para comparação. |
| `scripts/check_reference.py` | Oráculo das equações para conferir 36 pontos do histórico C. Não participa do experimento. |
| `scripts/analyze.py` | Reavalia as soluções, calcula estatísticas, NEH separada e gráficos. |
| `data/` | Matrizes das instâncias, em formato máquina × tarefa. |
| `results/` | Resultados reais, curvas por execução, agregações, resumos e gráficos. |
| `docs/RELATORIO.md` | Protocolo, resultados, interpretação e limites da avaliação. |
| `docs/VALIDACAO.md` | Evidências de corretude e ambiente utilizado. |

## Representação e avaliação do PFS

Cada posição `x[j]` é a chave real da tarefa `j`. A ordenação crescente das chaves
produz uma permutação completa. Empates são resolvidos pelo menor ID da tarefa.
Exemplo: `[0.5, -0.5, 0.5, 0.1]` resulta em `[2, 4, 1, 3]` na numeração externa.
Internamente os índices começam em zero; os CSVs mostram tarefas de 1 a 20.

Cada tarefa passa por todas as máquinas na mesma ordem. A sequência das tarefas
é igual em todas as máquinas; uma máquina processa uma operação por vez. As tarefas
estão disponíveis em tempo zero, sem setup, preempção, bloqueio ou restrição
no-wait. A espera entre máquinas é permitida.

Para a tarefa na posição i e a máquina k:

```text
C(i,k) = max(C(i-1,k), C(i,k-1)) + p[k][permutacao[i]]
C(0,k) = C(i,0) = 0
fitness = C(n,m) = makespan
```

`pfs_makespan` avalia em O(n*m), com O(m) memória; `pfs_decode` usa ordenação
estável por inserção, O(n²). Para as duas instâncias, n=20. Não se normaliza a
aptidão: o SMA recebe diretamente o makespan inteiro como `double`.

## Operadores e decisões numéricas

1. Inicialização independente de cada coordenada em `[LB, UB)`.
2. Em cada iteração `t=1,...,T`, projeta-se cada coordenada em `[LB,UB]` e avalia-se
   cada agente uma única vez.
3. Ordena-se a aptidão crescente. Para a posição k do ranking (base zero),
   `q=(S[k]-bF)/(wF-bF)`; `W=1+u*log10(1+q)` na primeira metade e
   `W=1-u*log10(1+q)` na segunda metade. Sorteio por coordenada. Se `wF==bF`,
   adota-se `q=0`, portanto `W=1`, como no documento. A demo MATLAB soma `eps`
   ao denominador; essa diferença numérica está explícita.
4. Se há melhora estrita, copia-se o vetor para o melhor histórico; essa cópia
   fica separada da matriz que será alterada.
5. `b=1-t/T`, `a=atanh(b)`, `p=tanh(abs(S[i]-DF))`.
6. Com probabilidade `z`, reinicializa-se o vetor usando **um único escalar**
   uniforme compartilhado entre as coordenadas, como a demo original.
7. Caso contrário, sorteiam-se os vetores `vb~U(-a,a)` e `vc~U(-b,b)`.
   Em cada coordenada sorteiam-se `r`, `A` e `B` (com reposição):

```text
se r < p: x[i,j] = best[j] + vb[j] * (W[i,j]*x[A,j] - x[B,j])
senão:    x[i,j] = vc[j] * x[i,j]
```

As atualizações são sequenciais na própria matriz. A e B podem coincidir e podem
ser o próprio agente. `S`, `W` e o melhor histórico permanecem fixos durante a
movimentação. Os sorteios de A e B ocorrem mesmo no ramo de contração.
A curva registra o melhor já **avaliado**, sem atribuir melhoria a posições
não avaliadas. A movimentação da iteração 500 não é avaliada, como em SMA.m.
Logo, são exatamente N*T chamadas ao objetivo, sem uma avaliação inicial extra.

Consequência importante da fidelidade: com limites iguais em todas as dimensões,
a reinicialização escalar produz chaves iguais. No PFS, isso decodifica sempre
para `[1,2,...,n]`. Não trocamos essa regra silenciosamente por uma reinicialização
vetorial. A adaptação pode perder exploração; mudar essa regra seria outra variante.
Os limites simétricos `[-1,1]` são uma escolha explícita desta aplicação, evitando
que a contração com `vc<0` seja sempre projetada em zero, como ocorreria em `[0,1]`.

## Parâmetros e origem

| Parâmetro | Valor | Origem |
|---|---|---|
| População N | 30 | `main.m` da demo dos autores e documento fornecido |
| Iterações T | 500 | `main.m` da demo dos autores e documento fornecido |
| Reinicialização z | 0,03 | `SMA.m` e documento fornecido |
| Repetições | 30 por instância | Requisito e protocolo do documento |
| Dimensão D | 20 | Uma chave por tarefa; substitui D=30 dos testes contínuos |
| LB, UB | -1, 1 | Decisão da adaptação por chaves reais, não padrão universal do PFS |
| ta001 | 20 × 5; semente 873654221 | Taillard / OR-Library |
| ta011 | 20 × 10; semente 587595453 | Taillard / OR-Library |
| Tempos | Inteiros em [1,99] | Gerador oficial Taillard |

As sementes do problema são fixas e diferentes das sementes do solver.
Os tempos de processamento são regenerados deterministicamente, na ordem
máquina → tarefa, com Park-Miller: `seed=(16807*seed)%2147483647`.
Não são instâncias aleatórias novas rotuladas como Taillard.

O solver usa SplitMix64 com estado `uint64_t`. A execução r de ta001 começa
em `202610190000+r`; ta011 começa em `202610290000+r`, para r=1,...,30.
São repetições pseudoaleatórias com estados iniciais distintos e sem compartilhar
população, melhor solução ou estado do RNG. Independência experimental não é
uma prova matemática de independência dos fluxos pseudoaleatórios.
Não se espera igualdade bit a bit com o MATLAB, que utiliza outro RNG.

## Dados de saída e estatística

- `history.csv`: 30.000 linhas de dados; instância, execução, semente, iteração,
  avaliações acumuladas e melhor makespan.
- `runs.csv`: 60 linhas; configuração, melhor solução final, tempo de CPU e permutação.
- `convergence_stats.csv`: 1.000 linhas; média e desvio padrão por iteração e instância.
- `summary.csv` / `summary.json`: melhores, piores, médias, medianas, desvios e comparação NEH.
- `convergencia.png` / `.svg`: duas curvas, cada uma calculada entre 30 execuções.

Para cada iteração t, a curva usa a média de `C_r(t)`; o desvio padrão amostral
é `sqrt(sum((C_r(t)-media(t))²)/(30-1))`. A região é média ± um DP, não intervalo
de confiança nem desvio dentro de uma população. As instâncias são agregadas
separadamente. Menor fitness é melhor.

## Resultados entregues

| Instância | Melhor | Média final | DP amostral | Mediana | Pior | NEH separada |
|---|---:|---:|---:|---:|---:|---:|
| ta001 | 1285 | 1296,97 | 3,77 | 1297 | 1305 | 1286 |
| ta011 | 1625 | 1663,97 | 25,53 | 1661 | 1742 | 1680 |

Em ta001 a média ficou 0,85% acima da NEH; em ta011 ficou 0,95% abaixo.
A NEH é uma solução viável de comparação e não o ótimo certificado. Não é um
experimento com orçamentos equivalentes entre algoritmos, nem demonstra superioridade
geral. O relatório apresenta os resultados sem esconder estagnação ou dispersão.

## Referências

1. Li, S.; Chen, H.; Wang, M.; Heidari, A. A.; Mirjalili, S. **Slime mould algorithm:
   A new method for stochastic optimization**. Future Generation Computer Systems,
   111, 300–323, 2020. https://doi.org/10.1016/j.future.2020.03.055
2. Autores do SMA. **SMA.m e main.m, demo versão 1.0 (2020)**. Página e pacote
   consultados em 19/10/2026: https://aliasgharheidari.com/SMA.html
   O arquivo atual do site contém vários algoritmos; foram conferidos somente
   os arquivos da pasta `Slime mould algorithm (SMA)-2020`.
3. Taillard, E. **Benchmarks for basic scheduling problems**. European Journal of
   Operational Research, 64(2), 278–285, 1993.
   https://doi.org/10.1016/0377-2217(93)90182-M
4. OR-Library (J. E. Beasley). Dados e gerador de Taillard, contribuição de
   D. C. Mattfeld e R. J. M. Vaessens, versão de 02/07/1996:
   https://people.brunel.ac.uk/~mastjjb/jeb/orlib/flowshopinfo.html
   https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/flowshop2.txt
   Consultado em 19/10/2026.
5. Nawaz, M.; Enscore Jr., E. E.; Ham, I. **A heuristic algorithm for the m-machine,
   n-job flow-shop sequencing problem**. Omega, 11(1), 91–95, 1983.
   https://doi.org/10.1016/0305-0483(83)90088-9
6. Dias, Wendson Menezes. **Slime Mould Algorithm: formulação matemática,
   representação e configuração dos testes**. Documento fornecido na solicitação,
   arquivo `sitemas_inteligentes-1(2).pdf`, 4 páginas.
