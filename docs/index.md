# Classificação da Faixa de Renda de Adultos

**Etapa 1: Análise Exploratória de Dados (EDA) e pré-processamento**

| | |
|---|---|
| **Aluno** | Leonardo Moretti Alva |
| **Data de entrega** | 13/09/2026 |
| **Dataset** | [Adult Income Census](https://www.kaggle.com/datasets/anaghakp/adult-income-census), Kaggle |
| **Notebook completo** | [EDA e pré-processamento](projeto_adult_income_eda.ipynb), com código, figuras e interpretações |
| **Arquivo `.ipynb`** | [Ver no GitHub](https://github.com/LeonardoMA07/projeto-classificacao-adult-income/blob/main/docs/projeto_adult_income_eda.ipynb) |

## Objetivo

Prever se a renda anual de uma pessoa é maior que 50 mil dólares (`>50K`) ou não (`<=50K`) a partir de dados demográficos e profissionais do Censo dos EUA de 1994. É um problema de classificação binária.

Esta etapa cobre a análise exploratória e a construção do pipeline de pré-processamento. A modelagem fica para a etapa seguinte.

## Os dados

- 31.947 registros e 11 features: 3 numéricas (`age`, `fnlwgt`, `education_num`) e 8 categóricas (`workclass`, `education`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country`).
- Alvo desbalanceado na proporção 3:1: 76% das pessoas ganham `<=50K`.
- Valores ausentes, marcados com `?`, em `workclass` e `occupation` (cerca de 5,6% cada) e `native_country` (25 linhas). Em `workclass` e `occupation` faltam nas mesmas linhas e marcam pessoas sem vínculo de trabalho.
- Inconsistência: 1.462 linhas trazem o texto "occupation" no lugar da ocupação. Esse grupo tem cerca de 53% de renda alta, o dobro da base, e por isso foi mantido como categoria própria.
- 65 linhas duplicadas, removidas.

## Principais achados

| Achado | Evidência |
|---|---|
| A escolaridade é a numérica mais ligada à renda | Correlação de 0,34 com o alvo; a taxa de `>50K` vai de 16% em *HS-grad* para 74% em *Doctorate* |
| Idade também importa | Correlação de 0,23; mediana de 44 anos em `>50K` contra 34 em `<=50K` |
| Estado civil separa bem as classes | 45% dos casados ganham `>50K`, contra 5% dos solteiros; parte do efeito vem da idade |
| `fnlwgt` é irrelevante | Correlação de -0,01 com a renda; é o peso amostral do Censo, não uma característica da pessoa |
| `education` e `education_num` são redundantes | Cada nível de escolaridade corresponde a exatamente um código |
| Viés de sexo | 31% dos homens ganham `>50K`, contra 11% das mulheres, mesmo com escolaridade mediana maior entre as mulheres |
| As numéricas separam só parcialmente as classes | No PCA, cada componente explica cerca de um terço da variância e as classes se sobrepõem |

## Estratégia de pré-processamento

| Etapa | Decisão | Motivo |
|---|---|---|
| Remoção de features | Descartar `education` e `fnlwgt` | Redundância e ausência de relação com o alvo |
| Valores ausentes | Categoria `Desconhecido` | A ausência carrega informação: só 10% desse grupo tem renda alta. A moda daria um emprego a quem não tem |
| Outliers | Manter todos | São idosos e níveis baixos de escolaridade, grupos reais |
| Encoding | One-Hot Encoding | Categorias sem ordem; *label encoding* criaria uma ordem falsa |
| Escala | `StandardScaler` | Coloca `age` e `education_num` na mesma escala para os modelos sensíveis a distância |

Tudo fica dentro de um `Pipeline` com `ColumnTransformer` do scikit-learn, ajustado só no conjunto de treino (80/20, estratificado). A saída tem 89 colunas.

## Recomendações para a etapa de modelagem

- Comparar Regressão Logística, Árvore de Decisão e Random Forest com validação cruzada estratificada.
- Avaliar com precisão, recall, F1-score e ROC-AUC, porque a acurácia engana com classes 3:1.
- Usar `class_weight='balanced'` e comparar as métricas entre homens e mulheres, por causa do viés encontrado.

## Como reproduzir

```bash
git clone https://github.com/LeonardoMA07/projeto-classificacao-adult-income.git
cd projeto-classificacao-adult-income
pip install -r requirements.txt
jupyter notebook docs/projeto_adult_income_eda.ipynb
```

O arquivo `adult income1.csv` já está na pasta `docs/`, ao lado do notebook. Basta executar todas as células.

## Estrutura do repositório

```
docs/
  index.md                        # esta página
  projeto_adult_income_eda.ipynb  # notebook com toda a análise
  adult income1.csv               # dataset do Kaggle
mkdocs.yml                        # configuração deste site
requirements.txt                  # bibliotecas usadas
```

## Referências

- Becker, B. e Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5XW20](https://doi.org/10.24432/C5XW20)
- [Adult Income Census](https://www.kaggle.com/datasets/anaghakp/adult-income-census), Kaggle.
- Documentação do [Pandas](https://pandas.pydata.org/docs/), [Scikit-learn](https://scikit-learn.org/stable/), [Seaborn](https://seaborn.pydata.org/) e [Matplotlib](https://matplotlib.org/stable/).
